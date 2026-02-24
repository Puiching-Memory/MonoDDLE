/**
 * CUDA kernel for rotated BEV IoU computation — float32.
 * Matches the original rotate_iou.py which uses np.float32 for computation.
 * The result is returned as float32, matching `boxes.astype(np.float32)` in
 * the original `rotate_iou_gpu_eval`.
 */
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_INT_PTS 16

__device__ float triangle_area_dev(const float* a, const float* b, const float* c) {
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0f;
}

__device__ float polygon_area(float* int_pts, int num_of_inter) {
    float area_val = 0.0f;
    for (int i = 0; i < num_of_inter - 2; i++) {
        area_val += fabsf(
            triangle_area_dev(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
    }
    return area_val;
}

__device__ void sort_vertex_in_convex_polygon_dev(float* int_pts, int num_of_inter) {
    if (num_of_inter <= 0) return;
    float center_x = 0.0f, center_y = 0.0f;
    for (int i = 0; i < num_of_inter; i++) {
        center_x += int_pts[2 * i];
        center_y += int_pts[2 * i + 1];
    }
    center_x /= num_of_inter;
    center_y /= num_of_inter;

    float vs[MAX_INT_PTS];
    for (int i = 0; i < num_of_inter; i++) {
        float vx = int_pts[2 * i] - center_x;
        float vy = int_pts[2 * i + 1] - center_y;
        // Match numpy NEP 50: math.sqrt(float32) computes in float64,
        // result cast to float32 when used in float32 division (weak type).
        float d = (float)sqrt((double)(vx * vx + vy * vy));
        if (d > 1e-8f) {
            vx /= d;
            vy /= d;
        }
        if (vy < 0) vx = -2.0f - vx;
        vs[i] = vx;
    }
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

__device__ bool line_segment_intersection_dev(
    const float* pts1, const float* pts2, int i, int j, float* temp_pts) {
    float A0 = pts1[2 * i], A1 = pts1[2 * i + 1];
    float B0 = pts1[2 * ((i + 1) % 4)], B1 = pts1[2 * ((i + 1) % 4) + 1];
    float C0 = pts2[2 * j], C1 = pts2[2 * j + 1];
    float D0 = pts2[2 * ((j + 1) % 4)], D1 = pts2[2 * ((j + 1) % 4) + 1];

    float BA0 = B0 - A0, BA1 = B1 - A1;
    float DA0 = D0 - A0, CA0 = C0 - A0;
    float DA1 = D1 - A1, CA1 = C1 - A1;

    bool acd = DA1 * CA0 > CA1 * DA0;
    bool bcd = (D1 - B1) * (C0 - B0) > (C1 - B1) * (D0 - B0);
    if (acd != bcd) {
        bool abc = CA1 * BA0 > BA1 * CA0;
        bool abd = DA1 * BA0 > BA1 * DA0;
        if (abc != abd) {
            float DC0 = D0 - C0, DC1 = D1 - C1;
            float ABBA = A0 * B1 - B0 * A1;
            float CDDC = C0 * D1 - D0 * C1;
            float DH = BA1 * DC0 - BA0 * DC1;
            if (fabsf(DH) > 1e-8f) {
                temp_pts[0] = (ABBA * DC0 - BA0 * CDDC) / DH;
                temp_pts[1] = (ABBA * DC1 - BA1 * CDDC) / DH;
                return true;
            }
        }
    }
    return false;
}

__device__ bool point_in_quadrilateral_dev(float pt_x, float pt_y, const float* corners) {
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

__device__ int quadrilateral_intersection_dev(
    const float* pts1, const float* pts2, float* int_pts) {
    int num_of_inter = 0;
    for (int i = 0; i < 4; i++) {
        if (point_in_quadrilateral_dev(pts1[2 * i], pts1[2 * i + 1], pts2)) {
            int_pts[num_of_inter * 2] = pts1[2 * i];
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
            num_of_inter++;
        }
        if (point_in_quadrilateral_dev(pts2[2 * i], pts2[2 * i + 1], pts1)) {
            int_pts[num_of_inter * 2] = pts2[2 * i];
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
            num_of_inter++;
        }
    }
    float temp_pts[2];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (line_segment_intersection_dev(pts1, pts2, i, j, temp_pts)) {
                int_pts[num_of_inter * 2] = temp_pts[0];
                int_pts[num_of_inter * 2 + 1] = temp_pts[1];
                num_of_inter++;
            }
        }
    }
    return num_of_inter;
}

__device__ void rbbox_to_corners_dev(float* corners, const float* rbbox) {
    // Match numpy NEP 50: math.cos(float32) computes in float64, result cast to
    // float32 when multiplied with float32 (Python scalar is "weak" type).
    float a_cos = (float)cos((double)rbbox[4]);
    float a_sin = (float)sin((double)rbbox[4]);
    float cx = rbbox[0], cy = rbbox[1];
    float xd = rbbox[2], yd = rbbox[3];

    float corners_x[4] = {-xd / 2.0f, -xd / 2.0f, xd / 2.0f, xd / 2.0f};
    float corners_y[4] = {-yd / 2.0f, yd / 2.0f, yd / 2.0f, -yd / 2.0f};

    for (int i = 0; i < 4; i++) {
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + cx;
        corners[2 * i + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + cy;
    }
}

__device__ float inter_dev(const float* rbbox1, const float* rbbox2) {
    float corners1[8], corners2[8], int_corners[32];
    rbbox_to_corners_dev(corners1, rbbox1);
    rbbox_to_corners_dev(corners2, rbbox2);
    int num = quadrilateral_intersection_dev(corners1, corners2, int_corners);
    sort_vertex_in_convex_polygon_dev(int_corners, num);
    return polygon_area(int_corners, num);
}

__device__ float devRotateIoUEval_dev(const float* rbox1, const float* rbox2, int criterion) {
    float area1 = rbox1[2] * rbox1[3];
    float area2 = rbox2[2] * rbox2[3];
    float area_inter = inter_dev(rbox1, rbox2);
    if (criterion == -1)
        return area_inter / (area1 + area2 - area_inter + 1e-8f);
    else if (criterion == 0)
        return area_inter / (area1 + 1e-8f);
    else if (criterion == 1)
        return area_inter / (area2 + 1e-8f);
    else
        return area_inter;
}

__global__ void rotate_iou_kernel(
    const float* boxes, const float* query_boxes,
    float* iou, int N, int K, int criterion) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int i = idx / K;
    int j = idx % K;
    iou[i * K + j] = devRotateIoUEval_dev(boxes + i * 5, query_boxes + j * 5, criterion);
}

/**
 * Rotate IoU GPU evaluation — float32 (matches original numpy behavior).
 * boxes:       [N, 5] (cx, cy, w, h, angle) — will be cast to float32
 * query_boxes: [K, 5]
 * Returns:     [N, K] IoU matrix (float32, matching original dtype)
 */
torch::Tensor rotate_iou_gpu(torch::Tensor boxes, torch::Tensor query_boxes, int criterion) {
    // Match original: boxes = boxes.astype(np.float32)
    boxes = boxes.contiguous().to(torch::kFloat32).to(torch::kCUDA);
    query_boxes = query_boxes.contiguous().to(torch::kFloat32).to(torch::kCUDA);

    int N = boxes.size(0);
    int K = query_boxes.size(0);

    auto iou = torch::zeros({N, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    if (N == 0 || K == 0) return iou.cpu();

    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    rotate_iou_kernel<<<blocks, threads>>>(
        boxes.data_ptr<float>(),
        query_boxes.data_ptr<float>(),
        iou.data_ptr<float>(),
        N, K, criterion);

    // Return as float32 matching original: iou.astype(boxes.dtype) where dtype=float32
    return iou.cpu();
}
