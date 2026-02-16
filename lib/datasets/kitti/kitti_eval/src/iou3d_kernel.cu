/*
 * KITTI 评估 CUDA 内核
 * ====================
 *
 * 精确复刻 kitti_eval_python/rotate_iou.py 的算法逻辑，
 * 保证计算结果与 numba CUDA 参考实现完全一致。
 *
 * 包含:
 *   - 2D 框 IoU (axis-aligned)
 *   - 旋转框 BEV IoU (基于点-四边形测试 + 线段相交)
 *   - 3D 框 IoU (BEV 交叠面积 × 高度交叠)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define THREADS_PER_BLOCK 64  // 8*8, 与参考实现一致

/* ═══════════════════ 旋转框 BEV IoU (精确复刻参考实现) ═══════════════════ */

/*
 * 三角形面积 (带符号)
 * 对应 rotate_iou.py: trangle_area(a, b, c)
 */
__device__ __forceinline__ float trangle_area(
    float ax, float ay, float bx, float by, float cx, float cy
) {
    return ((ax - cx) * (by - cy) - (ay - cy) * (bx - cx)) / 2.0f;
}

/*
 * 凸多边形面积 (三角扇)
 * 对应 rotate_iou.py: area(int_pts, num_of_inter)
 */
__device__ __forceinline__ float polygon_area_fan(const float* int_pts, int num_of_inter) {
    float area_val = 0.0f;
    for (int i = 0; i < num_of_inter - 2; i++) {
        area_val += fabsf(
            trangle_area(
                int_pts[0], int_pts[1],
                int_pts[2 * i + 2], int_pts[2 * i + 3],
                int_pts[2 * i + 4], int_pts[2 * i + 5]
            )
        );
    }
    return area_val;
}

/*
 * 凸多边形顶点排序 (极角插入排序)
 * 对应 rotate_iou.py: sort_vertex_in_convex_polygon(int_pts, num_of_inter)
 */
__device__ void sort_vertex_in_convex_polygon(float* int_pts, int num_of_inter) {
    if (num_of_inter <= 0) return;

    float center_x = 0.0f, center_y = 0.0f;
    for (int i = 0; i < num_of_inter; i++) {
        center_x += int_pts[2 * i];
        center_y += int_pts[2 * i + 1];
    }
    center_x /= num_of_inter;
    center_y /= num_of_inter;

    float vs[16];  // 最多 8 个交点
    for (int i = 0; i < num_of_inter; i++) {
        float vx = int_pts[2 * i] - center_x;
        float vy = int_pts[2 * i + 1] - center_y;
        float d = sqrtf(vx * vx + vy * vy);
        vx = vx / d;
        vy = vy / d;
        if (vy < 0) {
            vx = -2.0f - vx;
        }
        vs[i] = vx;
    }

    // 插入排序
    for (int i = 1; i < num_of_inter; i++) {
        if (vs[i - 1] > vs[i]) {
            float temp = vs[i];
            float tx = int_pts[2 * i];
            float ty = int_pts[2 * i + 1];
            int j = i;
            while (j > 0 && vs[j - 1] > temp) {
                vs[j] = vs[j - 1];
                int_pts[j * 2] = int_pts[j * 2 - 2];
                int_pts[j * 2 + 1] = int_pts[j * 2 - 1];
                j--;
            }
            vs[j] = temp;
            int_pts[j * 2] = tx;
            int_pts[j * 2 + 1] = ty;
        }
    }
}

/*
 * 线段相交检测
 * 对应 rotate_iou.py: line_segment_intersection(pts1, pts2, i, j, temp_pts)
 */
__device__ __forceinline__ bool line_segment_intersection(
    const float* pts1, const float* pts2,
    int i, int j,
    float* temp_pts
) {
    float Ax = pts1[2 * i];
    float Ay = pts1[2 * i + 1];
    float Bx = pts1[2 * ((i + 1) % 4)];
    float By = pts1[2 * ((i + 1) % 4) + 1];
    float Cx = pts2[2 * j];
    float Cy = pts2[2 * j + 1];
    float Dx = pts2[2 * ((j + 1) % 4)];
    float Dy = pts2[2 * ((j + 1) % 4) + 1];

    float BA0 = Bx - Ax;
    float BA1 = By - Ay;
    float DA0 = Dx - Ax;
    float CA0 = Cx - Ax;
    float DA1 = Dy - Ay;
    float CA1 = Cy - Ay;
    bool acd = DA1 * CA0 > CA1 * DA0;
    bool bcd = (Dy - By) * (Cx - Bx) > (Cy - By) * (Dx - Bx);
    if (acd != bcd) {
        bool abc = CA1 * BA0 > BA1 * CA0;
        bool abd = DA1 * BA0 > BA1 * DA0;
        if (abc != abd) {
            float DC0 = Dx - Cx;
            float DC1 = Dy - Cy;
            float ABBA = Ax * By - Bx * Ay;
            float CDDC = Cx * Dy - Dx * Cy;
            float DH = BA1 * DC0 - BA0 * DC1;
            float Ddx = ABBA * DC0 - BA0 * CDDC;
            float Ddy = ABBA * DC1 - BA1 * CDDC;
            temp_pts[0] = Ddx / DH;
            temp_pts[1] = Ddy / DH;
            return true;
        }
    }
    return false;
}

/*
 * 点在四边形内检测
 * 对应 rotate_iou.py: point_in_quadrilateral(pt_x, pt_y, corners)
 */
__device__ __forceinline__ bool point_in_quadrilateral(
    float pt_x, float pt_y, const float* corners
) {
    float ab0 = corners[2] - corners[0];
    float ab1 = corners[3] - corners[1];
    float ad0 = corners[6] - corners[0];
    float ad1 = corners[7] - corners[1];
    float ap0 = pt_x - corners[0];
    float ap1 = pt_y - corners[1];

    float abab = ab0 * ab0 + ab1 * ab1;
    float abap = ab0 * ap0 + ab1 * ap1;
    float adad = ad0 * ad0 + ad1 * ad1;
    float adap = ad0 * ap0 + ad1 * ap1;

    return abab >= abap && abap >= 0 && adad >= adap && adap >= 0;
}

/*
 * 四边形交叠顶点收集
 * 对应 rotate_iou.py: quadrilateral_intersection(pts1, pts2, int_pts)
 */
__device__ int quadrilateral_intersection(
    const float* pts1, const float* pts2, float* int_pts
) {
    int num_of_inter = 0;
    for (int i = 0; i < 4; i++) {
        if (point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2)) {
            int_pts[num_of_inter * 2] = pts1[2 * i];
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
            num_of_inter++;
        }
        if (point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1)) {
            int_pts[num_of_inter * 2] = pts2[2 * i];
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
            num_of_inter++;
        }
    }
    float temp_pts[2];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (line_segment_intersection(pts1, pts2, i, j, temp_pts)) {
                int_pts[num_of_inter * 2] = temp_pts[0];
                int_pts[num_of_inter * 2 + 1] = temp_pts[1];
                num_of_inter++;
            }
        }
    }
    return num_of_inter;
}

/*
 * 旋转框 → 4 个顶点 (顺时针)
 * 对应 rotate_iou.py: rbbox_to_corners(corners, rbbox)
 */
__device__ __forceinline__ void rbbox_to_corners(float* corners, const float* rbbox) {
    float angle = rbbox[4];
    float a_cos = cosf(angle);
    float a_sin = sinf(angle);
    float center_x = rbbox[0];
    float center_y = rbbox[1];
    float x_d = rbbox[2];
    float y_d = rbbox[3];

    float corners_x[4] = {-x_d / 2, -x_d / 2, x_d / 2, x_d / 2};
    float corners_y[4] = {-y_d / 2, y_d / 2, y_d / 2, -y_d / 2};

    for (int i = 0; i < 4; i++) {
        corners[2 * i]     = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x;
        corners[2 * i + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y;
    }
}

/*
 * 两个旋转框的交叠面积
 * 对应 rotate_iou.py: inter(rbbox1, rbbox2)
 */
__device__ __forceinline__ float inter_area(const float* rbbox1, const float* rbbox2) {
    float corners1[8], corners2[8];
    float intersection_corners[16];  // 最多 8 个交点

    rbbox_to_corners(corners1, rbbox1);
    rbbox_to_corners(corners2, rbbox2);

    int num_intersection = quadrilateral_intersection(corners1, corners2, intersection_corners);
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection);

    return polygon_area_fan(intersection_corners, num_intersection);
}

/*
 * 旋转框 IoU (带 criterion 参数)
 * 对应 rotate_iou.py: devRotateIoUEval(rbox1, rbox2, criterion)
 *
 * criterion:
 *   -1: IoU = intersection / union
 *    0: intersection / area1
 *    1: intersection / area2
 *    2: 仅返回交叠面积
 */
__device__ __forceinline__ float devRotateIoUEval(
    const float* rbox1, const float* rbox2, int criterion
) {
    float area1 = rbox1[2] * rbox1[3];
    float area2 = rbox2[2] * rbox2[3];
    float area_intersection = inter_area(rbox1, rbox2);

    if (criterion == -1) {
        float ua = area1 + area2 - area_intersection;
        return (ua > 0.0f) ? area_intersection / ua : 0.0f;
    } else if (criterion == 0) {
        return (area1 > 0.0f) ? area_intersection / area1 : 0.0f;
    } else if (criterion == 1) {
        return (area2 > 0.0f) ? area_intersection / area2 : 0.0f;
    } else {
        return area_intersection;
    }
}

/* ═══════════════════ CUDA Kernels ═══════════════════ */

/*
 * 旋转框 BEV IoU 内核
 * 对应 rotate_iou.py: rotate_iou_kernel_eval
 *
 * 使用共享内存分块处理，与参考实现的 block 策略完全一致。
 * 输入: dev_boxes (N*5), dev_query_boxes (K*5), 格式: [cx, cy, w, h, angle]
 * 输出: dev_iou (N*K), 行 = boxes, 列 = query_boxes
 */
__global__ void rotate_iou_kernel_eval(
    int64_t N, int64_t K,
    const float* dev_boxes,
    const float* dev_query_boxes,
    float* dev_iou,
    int criterion
) {
    const int threadsPerBlock = THREADS_PER_BLOCK;
    int row_start = blockIdx.x;
    int col_start = blockIdx.y;
    int tx = threadIdx.x;

    int row_size = min((int64_t)threadsPerBlock, N - (int64_t)row_start * threadsPerBlock);
    int col_size = min((int64_t)threadsPerBlock, K - (int64_t)col_start * threadsPerBlock);

    __shared__ float block_boxes[THREADS_PER_BLOCK * 5];
    __shared__ float block_qboxes[THREADS_PER_BLOCK * 5];

    int dev_query_box_idx = threadsPerBlock * col_start + tx;
    int dev_box_idx = threadsPerBlock * row_start + tx;

    if (tx < col_size) {
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0];
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1];
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2];
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3];
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4];
    }
    if (tx < row_size) {
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0];
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1];
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2];
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3];
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4];
    }
    __syncthreads();

    if (tx < row_size) {
        for (int i = 0; i < col_size; i++) {
            int64_t offset = (int64_t)row_start * threadsPerBlock * K
                           + (int64_t)col_start * threadsPerBlock
                           + (int64_t)tx * K + i;
            // 注意: 参考实现调用 devRotateIoUEval(qbox, box, ...)
            // 即 rbox1 = query_box, rbox2 = box
            dev_iou[offset] = devRotateIoUEval(
                &block_qboxes[i * 5],
                &block_boxes[tx * 5],
                criterion
            );
        }
    }
}

/*
 * 2D 轴对齐框 IoU 内核
 * 对应 eval.py: image_box_overlap
 */
__global__ void iou2d_kernel(
    int N, int K,
    const float* boxes, const float* query_boxes,
    float* out, int criterion
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    int n = idx / K;
    int k = idx % K;

    float b_x1 = boxes[n * 4 + 0];
    float b_y1 = boxes[n * 4 + 1];
    float b_x2 = boxes[n * 4 + 2];
    float b_y2 = boxes[n * 4 + 3];

    float q_x1 = query_boxes[k * 4 + 0];
    float q_y1 = query_boxes[k * 4 + 1];
    float q_x2 = query_boxes[k * 4 + 2];
    float q_y2 = query_boxes[k * 4 + 3];

    float iw = fminf(b_x2, q_x2) - fmaxf(b_x1, q_x1);
    if (iw <= 0) { out[idx] = 0.0f; return; }
    float ih = fminf(b_y2, q_y2) - fmaxf(b_y1, q_y1);
    if (ih <= 0) { out[idx] = 0.0f; return; }

    float inter = iw * ih;
    float qbox_area = (q_x2 - q_x1) * (q_y2 - q_y1);

    if (criterion == -1) {
        float ua = (b_x2 - b_x1) * (b_y2 - b_y1) + qbox_area - inter;
        out[idx] = (ua > 0.0f) ? inter / ua : 0.0f;
    } else if (criterion == 0) {
        float box_area = (b_x2 - b_x1) * (b_y2 - b_y1);
        out[idx] = (box_area > 0.0f) ? inter / box_area : 0.0f;
    } else if (criterion == 1) {
        out[idx] = (qbox_area > 0.0f) ? inter / qbox_area : 0.0f;
    } else {
        out[idx] = inter;
    }
}

/*
 * 3D 框 IoU 内核
 * 对应 eval.py: d3_box_overlap_kernel + d3_box_overlap
 *
 * 输入框格式: [x, y, z, l, h, w, ry] (7 个元素)
 *   - (x, z) 为 BEV 坐标
 *   - l, w 为 BEV 尺寸
 *   - y 为高度方向最大值 (底部)
 *   - h 为高度维度
 *   - ry 为绕 Y 轴旋转角
 *
 * 算法:
 *   1. 提取 BEV 参数: [x, z, l, w, ry]
 *   2. 计算 BEV 交叠面积 (criterion=2)
 *   3. 计算高度交叠 iw = min(y1, y2) - max(y1-h1, y2-h2)
 *   4. 3D 交叠 = BEV交叠 × 高度交叠
 *   5. 按 criterion 计算最终值
 */
__global__ void iou3d_kernel(
    int N, int K,
    const float* boxes, const float* query_boxes,
    float* out, int criterion
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    int i = idx / K;
    int j = idx % K;

    // 提取 BEV 参数: [x, z, l, w, ry]
    float box_bev[5] = {
        boxes[i * 7 + 0],      // x
        boxes[i * 7 + 2],      // z
        boxes[i * 7 + 3],      // l (对应 dimensions[:, 0])
        boxes[i * 7 + 5],      // w (对应 dimensions[:, 2])
        boxes[i * 7 + 6]       // ry
    };

    float q_bev[5] = {
        query_boxes[j * 7 + 0],
        query_boxes[j * 7 + 2],
        query_boxes[j * 7 + 3],
        query_boxes[j * 7 + 5],
        query_boxes[j * 7 + 6]
    };

    // BEV 交叠面积 (criterion=2 → 仅面积)
    float bev_inter = devRotateIoUEval(box_bev, q_bev, 2);

    // 高度交叠
    float b_y_max = boxes[i * 7 + 1];
    float b_h = boxes[i * 7 + 4];
    float q_y_max = query_boxes[j * 7 + 1];
    float q_h = query_boxes[j * 7 + 4];

    float iw = fminf(b_y_max, q_y_max) - fmaxf(b_y_max - b_h, q_y_max - q_h);

    float intersection_3d = 0.0f;
    if (iw > 0 && bev_inter > 0) {
        intersection_3d = bev_inter * iw;
    }

    // 按 criterion 计算
    if (intersection_3d <= 0.0f) {
        out[idx] = 0.0f;
        return;
    }

    float area1 = boxes[i * 7 + 3] * boxes[i * 7 + 4] * boxes[i * 7 + 5];
    float area2 = query_boxes[j * 7 + 3] * query_boxes[j * 7 + 4] * query_boxes[j * 7 + 5];

    if (criterion == -1) {
        float ua = area1 + area2 - intersection_3d;
        out[idx] = (ua > 0.0f) ? intersection_3d / ua : 0.0f;
    } else if (criterion == 0) {
        out[idx] = (area1 > 0.0f) ? intersection_3d / area1 : 0.0f;
    } else if (criterion == 1) {
        out[idx] = (area2 > 0.0f) ? intersection_3d / area2 : 0.0f;
    } else {
        out[idx] = intersection_3d;
    }
}


/* ═══════════════════ C++ 调用接口 ═══════════════════ */

void launch_rotate_iou_kernel(
    int64_t N, int64_t K,
    const float* boxes, const float* query_boxes,
    float* iou, int criterion
) {
    if (N == 0 || K == 0) return;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK), DIVUP(K, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    rotate_iou_kernel_eval<<<blocks, threads>>>(N, K, boxes, query_boxes, iou, criterion);
    cudaDeviceSynchronize();
}

void launch_iou2d_kernel(
    int N, int K,
    const float* boxes, const float* query_boxes,
    float* out, int criterion
) {
    if (N == 0 || K == 0) return;
    int threads = 1024;
    int blocks = DIVUP(N * K, threads);
    iou2d_kernel<<<blocks, threads>>>(N, K, boxes, query_boxes, out, criterion);
    cudaDeviceSynchronize();
}

void launch_iou3d_kernel(
    int N, int K,
    const float* boxes, const float* query_boxes,
    float* out, int criterion
) {
    if (N == 0 || K == 0) return;
    int threads = 1024;
    int blocks = DIVUP(N * K, threads);
    iou3d_kernel<<<blocks, threads>>>(N, K, boxes, query_boxes, out, criterion);
    cudaDeviceSynchronize();
}
