#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

const float EPS = 1e-8;

__device__ inline float check_rect_cross(const float& p1x, const float& p1y, const float& p2x, const float& p2y, const float& q1x, const float& q1y, const float& q2x, const float& q2y, float& inter_x, float& inter_y) {
    float dx1 = p2x - p1x;
    float dy1 = p2y - p1y;
    float dx2 = q2x - q1x;
    float dy2 = q2y - q1y;
    
    float D = dx1 * dy2 - dy1 * dx2;
    if (abs(D) < 1e-8) return 0.0f;
    
    float D1 = (q1x - p1x) * dy2 - (q1y - p1y) * dx2;
    float D2 = (q1x - p1x) * dy1 - (q1y - p1y) * dx1;
    
    float t1 = D1 / D;
    float t2 = D2 / D;
    
    if (t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1) {
        inter_x = p1x + t1 * dx1;
        inter_y = p1y + t1 * dy1;
        return 1.0f;
    }
    return 0.0f;
}

__device__ inline float polygon_area(const float* poly_x, const float* poly_y, int count) {
    if (count < 3) return 0.0f;
    float area = 0.0f;
    for (int i = 0; i < count; i++) {
        area += poly_x[i] * poly_y[(i + 1) % count] - poly_x[(i + 1) % count] * poly_y[i];
    }
    return 0.5f * abs(area);
}


__device__ inline float devRotateIoU(const float* box_a, const float* box_b) {
    // params: box_a (5): x, y, w, h, angle (rad/deg?) - Python code does conversion. 
    // We assume input here is x, y, w, h, angle (radians).
    
    // Unpack
    float cx1 = box_a[0];
    float cy1 = box_a[1];
    float w1 = box_a[2];
    float h1 = box_a[3];
    float a1 = box_a[4]; // expected in radians

    float cx2 = box_b[0];
    float cy2 = box_b[1];
    float w2 = box_b[2];
    float h2 = box_b[3];
    float a2 = box_b[4]; // radians

    float cos1 = cos(a1), sin1 = sin(a1);
    float cos2 = cos(a2), sin2 = sin(a2);

    float rect1_x[4], rect1_y[4];
    float rect2_x[4], rect2_y[4];
    
    // Generate vertices for rect1 (centered)
    // 0: -w/2, -h/2 -> (-w/2)*c - (-h/2)*s + cx
    // Order: (- -), (+ -), (+ +), (- +)
    // 0: bottom-left, 1: bottom-right, 2: top-right, 3: top-left (in standard coords)
    
    float dx1[4] = {-w1/2, w1/2, w1/2, -w1/2};
    float dy1[4] = {-h1/2, -h1/2, h1/2, h1/2};
    
    for (int i=0; i<4; i++) {
        rect1_x[i] = cx1 + dx1[i]*cos1 - dy1[i]*sin1;
        rect1_y[i] = cy1 + dx1[i]*sin1 + dy1[i]*cos1;
    }
    
    float dx2[4] = {-w2/2, w2/2, w2/2, -w2/2};
    float dy2[4] = {-h2/2, -h2/2, h2/2, h2/2};
    
    for (int i=0; i<4; i++) {
        rect2_x[i] = cx2 + dx2[i]*cos2 - dy2[i]*sin2;
        rect2_y[i] = cy2 + dx2[i]*sin2 + dy2[i]*cos2;
    }
    
    // Sutherland-Hodgman clipping
    // Max vertices = 8
    float poly_x[24], poly_y[24];
    float new_poly_x[24], new_poly_y[24];
    int count = 4;
    for(int i=0; i<4; i++) {
        poly_x[i] = rect1_x[i];
        poly_y[i] = rect1_y[i];
    }
    
    // Clip against edges of rect2
    for(int edge=0; edge<4; edge++) {
        float edge_start_x = rect2_x[edge];
        float edge_start_y = rect2_y[edge];
        float edge_end_x = rect2_x[(edge+1)%4];
        float edge_end_y = rect2_y[(edge+1)%4];
        
        // Edge normal (inward pointing)
        // Edge vector = (dx, dy). Normal (-dy, dx)
        // Check center of rect2 to determine direction? 
        // Or simpler: just use line equation CP (Cross Product) > 0
        // (p.x - start.x) * (end.y - start.y) - (p.y - start.y) * (end.x - start.x)
        
        int new_count = 0;
        
        float cp_prev;
        {
             // Check last point
            float dx = edge_end_x - edge_start_x;
            float dy = edge_end_y - edge_start_y;
            float px = poly_x[count-1] - edge_start_x;
            float py = poly_y[count-1] - edge_start_y;
            cp_prev = px * dy - py * dx;
        }

        for(int i=0; i<count; i++) {
            float dx = edge_end_x - edge_start_x;
            float dy = edge_end_y - edge_start_y;
            float px = poly_x[i] - edge_start_x;
            float py = poly_y[i] - edge_start_y;
            float cp_curr = px * dy - py * dx;
            
            // Check sign. Standard counter-clockwise vertices have 'inside' as one sign.
            // Let's assume CCW. Inside is Left. CP > 0.
            // But let's verify orientation.
            // rect2 ordered (- -), (+ -) ... etc.
            // bottom-left -> bottom-right. dx > 0, dy = 0. Normal (0, 1). CP = px * 0 - py * dx = -py*dx.
            // if py > 0 (inside), CP < 0. So Inside is CP <= 0?
            // Wait, standard polygon area is positive if CCW.
            // Let's rely on vertices order.
            
            // Actually, simpler logic:
            // Point p is inside if (p-p1) dot (p2-p1)_orth >= 0 where orth points inwards.
            // Or just use the sign consistently.
            
            // If prev was 'inside' and curr is 'outside', add intersection.
            // If prev was 'outside' and curr is 'inside', add intersection and curr.
            // If both inside, add curr.
            
            // Let's use a simpler check since we know they are convex logic:
            // Just use infinite lines clipping.
            int prev_in = (cp_prev <= 0); // Assuming <=0 is inside given the vertex order (-w/2, -h/2)...
            int curr_in = (cp_curr <= 0);
            
            if (prev_in && curr_in) {
                 new_poly_x[new_count] = poly_x[i];
                 new_poly_y[new_count] = poly_y[i];
                 new_count++;
            } else if (!prev_in && curr_in) {
                // Intersection
                float t = cp_prev / (cp_prev - cp_curr);
                new_poly_x[new_count] = poly_x[count-1] + t * (poly_x[i] - poly_x[count-1]);
                new_poly_y[new_count] = poly_y[count-1] + t * (poly_y[i] - poly_y[count-1]);
                new_count++;
                new_poly_x[new_count] = poly_x[i];
                new_poly_y[new_count] = poly_y[i];
                new_count++;
            } else if (prev_in && !curr_in) {
                // Intersection
                float t = cp_prev / (cp_prev - cp_curr);
                new_poly_x[new_count] = poly_x[count-1] + t * (poly_x[i] - poly_x[count-1]);
                new_poly_y[new_count] = poly_y[count-1] + t * (poly_y[i] - poly_y[count-1]);
                new_count++;
            }
            
            cp_prev = cp_curr;
        }
        count = new_count;
        for(int k=0; k<count; k++) {
            poly_x[k] = new_poly_x[k];
            poly_y[k] = new_poly_y[k];
        }
    }
    
    return polygon_area(poly_x, poly_y, count);
}

__global__ void iou3d_kernel(int N, int K, const float* boxes, const float* query_boxes, float* out, int criterion) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    int i = idx / K;
    int j = idx % K;

    // Boxes: x, y, z, h, w, l, ry (7)
    // Indexes: x=0, y=1, z=2, h=3, w=4, l=5, ry=6
    // BUT we need to match python code usage:
    // BEV: x(0), z(2), h(3), l(5), ry(6)
    // Height: y(1), h(4) ?? No.
    // In d3_box_overlap: 
    //   iw = min(b_y, q_y) - max(b_y - b_h, q_y - q_h)
    //   where b_y = boxes[i, 1], b_h = boxes[i, 4]
    //   So y(1) is TOP (max y), h(4) is HEIGHT.
    
    // Fetch BEV params
    float box_bev[5];
    box_bev[0] = boxes[i * 7 + 0]; // x
    box_bev[1] = boxes[i * 7 + 2]; // z (y in BEV)
    box_bev[2] = boxes[i * 7 + 3]; // h (width in BEV) - Note: Using 3 as w
    box_bev[3] = boxes[i * 7 + 5]; // l (height in BEV) - Note: Using 5 as h
    
    // Angle conversion: Python side did: -angle * 180 / PI. 
    // And passed that to Triton.
    // If we assume `ry` is in radians in the input tensor.
    // Standard rotation matrix in `devRotateIoU` uses standard radians.
    // KITTI ry: rotation around Y-axis.
    // We need to check if 3 and 5 are w and h respectively or h and w.
    // Usually l is along the heading.
    // So if angle=0, box extends in l along X? Or Z?
    // In KITTI, ry=0 -> along X.
    // If standard rect in `devRotateIoU` assumes w along X, h along Y (before rotation).
    // Then we should map appropriate dimensions.
    
    // Let's assume the Python code logic regarding BEV is correct and replicate it.
    // Python helper `_format_rotated_boxes`:
    //   x = box[0], y = box[1] (here z), w = box[2] (here h), h = box[3] (here l)
    //   angle = -ry * 180 / pi.
    // Then Triton calculates cos/sin of (angle_deg * pi / 180) => cos(-ry).
    // So it effectively uses -ry for rotation.
    // cos(-ry) = cos(ry), sin(-ry) = -sin(ry).
    
    // Our devRotateIoU takes radians. So we pass -ry.
    box_bev[4] = -boxes[i * 7 + 6]; 

    float q_bev[5];
    q_bev[0] = query_boxes[j * 7 + 0];
    q_bev[1] = query_boxes[j * 7 + 2];
    q_bev[2] = query_boxes[j * 7 + 3];
    q_bev[3] = query_boxes[j * 7 + 5];
    q_bev[4] = -query_boxes[j * 7 + 6];
    
    float bev_inter_area = devRotateIoU(box_bev, q_bev);
    
    // Height overlap
    float b_y_max = boxes[i * 7 + 1];
    float b_h_dim = boxes[i * 7 + 4]; // The dimension used for height overlap
    float b_y_min = b_y_max - b_h_dim;
    
    float q_y_max = query_boxes[j * 7 + 1];
    float q_h_dim = query_boxes[j * 7 + 4];
    float q_y_min = q_y_max - q_h_dim;
    
    float iw = min(b_y_max, q_y_max) - max(b_y_min, q_y_min);
    
    float intersection_3d = 0.0f;
    if (iw > 0) {
        intersection_3d = bev_inter_area * iw;
    }
    
    // Now handle criterion
    // criterion == -1: IoU
    // criterion == 0: Intersection / Area1 (Self)
    // criterion == 1: Intersection / Area2 (Query)
    // criterion == 2: Intersection
    
    if (criterion == 2) {
        out[idx] = intersection_3d;
    } else {
        float area1 = boxes[i * 7 + 3] * boxes[i * 7 + 4] * boxes[i * 7 + 5];
        float area2 = query_boxes[j * 7 + 3] * query_boxes[j * 7 + 4] * query_boxes[j * 7 + 5];
        
        float ua = area1 + area2 - intersection_3d;
        
        if (criterion == -1) {
             out[idx] = (ua > 0) ? intersection_3d / ua : 0.0f;
        } else if (criterion == 0) {
             out[idx] = (area1 > 0) ? intersection_3d / area1 : 0.0f;
        } else if (criterion == 1) {
             out[idx] = (area2 > 0) ? intersection_3d / area2 : 0.0f;
        }
    }
}


// C++ Interface
void boxes_iou3d_gpu(at::Tensor boxes, at::Tensor query_boxes, at::Tensor out, int criterion) {
    CHECK_INPUT(boxes);
    CHECK_INPUT(query_boxes);
    CHECK_INPUT(out);
    
    int N = boxes.size(0);
    int K = query_boxes.size(0);
    
    int threads = 1024;
    int blocks = DIVUP(N * K, threads);
    
    iou3d_kernel<<<blocks, threads>>>(N, K, 
        boxes.data_ptr<float>(), 
        query_boxes.data_ptr<float>(), 
        out.data_ptr<float>(), 
        criterion);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in iou3d_kernel: %s\n", cudaGetErrorString(err));
    }
}

// Separate kernel for BEV IoU/Intersection Only (matches rotated_iou_gpu functionality but in CUDA)
__global__ void iou_bev_kernel(int N, int K, const float* boxes, const float* query_boxes, float* out, int criterion) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    int i = idx / K;
    int j = idx % K;
    
    // Boxes assumed to be (N, 5): x, y, w, h, angle
    // Just strict BEV calculation
    float box_bev[5];
    // Copy directly. Assumes user handled format already.
    box_bev[0] = boxes[i*5+0]; box_bev[1] = boxes[i*5+1]; box_bev[2] = boxes[i*5+2]; box_bev[3] = boxes[i*5+3]; 
    // Assumes input is radians. And applies sign flip to match KITTI/standard conversion if needed.
    // rotate_iou_gpu_eval converts rad clockwise to deg ccw (-angle).
    // so we pass -angle (rad) here.
    box_bev[4] = -boxes[i*5+4];

    float q_bev[5];
    q_bev[0] = query_boxes[j*5+0]; q_bev[1] = query_boxes[j*5+1]; q_bev[2] = query_boxes[j*5+2]; q_bev[3] = query_boxes[j*5+3]; 
    q_bev[4] = -query_boxes[j*5+4];
    
    float intersection = devRotateIoU(box_bev, q_bev);
    
    if (criterion == 2) {
        out[idx] = intersection;
    } else {
        float area1 = box_bev[2] * box_bev[3];
        float area2 = q_bev[2] * q_bev[3];
        float ua = area1 + area2 - intersection;
         if (criterion == -1) {
             out[idx] = (ua > 0) ? intersection / ua : 0.0f;
        } else if (criterion == 0) {
             out[idx] = (area1 > 0) ? intersection / area1 : 0.0f;
        } else if (criterion == 1) {
             out[idx] = (area2 > 0) ? intersection / area2 : 0.0f;
        }
    }
}

void boxes_iou_bev_gpu(at::Tensor boxes, at::Tensor query_boxes, at::Tensor out, int criterion) {
    CHECK_INPUT(boxes);
    CHECK_INPUT(query_boxes);
    CHECK_INPUT(out);
    
    int N = boxes.size(0);
    int K = query_boxes.size(0);
    
    int threads = 1024;
    int blocks = DIVUP(N * K, threads);
    
    iou_bev_kernel<<<blocks, threads>>>(N, K, 
        boxes.data_ptr<float>(), 
        query_boxes.data_ptr<float>(), 
        out.data_ptr<float>(), 
        criterion);
}

__global__ void iou2d_kernel(int N, int K, const float* boxes, const float* query_boxes, float* out, int criterion) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    int i = idx / K;
    int j = idx % K;

    // boxes: (N, 4) x1, y1, x2, y2
    // query_boxes: (K, 4)
    
    float b_x1 = boxes[i*4+0];
    float b_y1 = boxes[i*4+1];
    float b_x2 = boxes[i*4+2];
    float b_y2 = boxes[i*4+3];

    float q_x1 = query_boxes[j*4+0];
    float q_y1 = query_boxes[j*4+1];
    float q_x2 = query_boxes[j*4+2];
    float q_y2 = query_boxes[j*4+3];

    float iw = min(b_x2, q_x2) - max(b_x1, q_x1);
    float ih = min(b_y2, q_y2) - max(b_y1, q_y1);
    
    float inter = 0.0f;
    if (iw > 0 && ih > 0) {
        inter = iw * ih;
    }
    
    if (criterion == 2) {
        out[idx] = inter;
    } else {
        float area1 = (b_x2 - b_x1) * (b_y2 - b_y1);
        float area2 = (q_x2 - q_x1) * (q_y2 - q_y1);
        float ua = area1 + area2 - inter;
        
        if (criterion == -1) {
             out[idx] = (ua > 0) ? inter / ua : 0.0f;
        } else if (criterion == 0) {
             out[idx] = (area1 > 0) ? inter / area1 : 0.0f;
        } else if (criterion == 1) {
             out[idx] = (area2 > 0) ? inter / area2 : 0.0f;
        }
    }
}

void boxes_iou2d_gpu(at::Tensor boxes, at::Tensor query_boxes, at::Tensor out, int criterion) {
    CHECK_INPUT(boxes);
    CHECK_INPUT(query_boxes);
    CHECK_INPUT(out);
    
    int N = boxes.size(0);
    int K = query_boxes.size(0);
    
    int threads = 1024;
    int blocks = DIVUP(N * K, threads);
    
    iou2d_kernel<<<blocks, threads>>>(N, K, 
        boxes.data_ptr<float>(), 
        query_boxes.data_ptr<float>(), 
        out.data_ptr<float>(), 
        criterion);
}

