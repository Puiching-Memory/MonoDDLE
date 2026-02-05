#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

// Forward declaration
void boxes_iou3d_gpu(at::Tensor boxes, at::Tensor query_boxes, at::Tensor out, int criterion);
void boxes_iou_bev_gpu(at::Tensor boxes, at::Tensor query_boxes, at::Tensor out, int criterion);
void boxes_iou2d_gpu(at::Tensor boxes, at::Tensor query_boxes, at::Tensor out, int criterion);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor iou3d_gpu(at::Tensor boxes, at::Tensor query_boxes, int criterion) {
    CHECK_INPUT(boxes);
    CHECK_INPUT(query_boxes);
    
    int N = boxes.size(0);
    int K = query_boxes.size(0);
    
    auto out = torch::zeros({N, K}, boxes.options());
    
    boxes_iou3d_gpu(boxes, query_boxes, out, criterion);
    
    return out;
}

at::Tensor iou_bev_gpu(at::Tensor boxes, at::Tensor query_boxes, int criterion) {
    CHECK_INPUT(boxes);
    CHECK_INPUT(query_boxes);
    
    int N = boxes.size(0);
    int K = query_boxes.size(0);
    
    auto out = torch::zeros({N, K}, boxes.options());
    
    boxes_iou_bev_gpu(boxes, query_boxes, out, criterion);
    
    return out;
}

at::Tensor iou2d_gpu(at::Tensor boxes, at::Tensor query_boxes, int criterion) {
    CHECK_INPUT(boxes);
    CHECK_INPUT(query_boxes);
    
    int N = boxes.size(0);
    int K = query_boxes.size(0);
    
    auto out = torch::zeros({N, K}, boxes.options());
    
    boxes_iou2d_gpu(boxes, query_boxes, out, criterion);
    
    return out;
}

// Helper for sorting
struct Candidate {
    int index;
    float overlap;
    int ignored_det;
};

// C++ Implementation of compute_statistics_jit
std::vector<torch::Tensor> compute_statistics_cpp(
    at::Tensor overlaps,
    at::Tensor gt_datas,
    at::Tensor dt_datas,
    at::Tensor ignored_gt,
    at::Tensor ignored_det,
    at::Tensor dc_bboxes,
    int metric,
    float min_overlap,
    float thresh,
    bool compute_aos
) {
    // Inputs are expected to be on CPU for this function
    auto overlaps_a = overlaps.accessor<float, 2>();
    auto gt_datas_a = gt_datas.accessor<float, 2>();
    auto dt_datas_a = dt_datas.accessor<float, 2>();
    auto ignored_gt_a = ignored_gt.accessor<int64_t, 1>();
    auto ignored_det_a = ignored_det.accessor<int64_t, 1>();
    
    int gt_size = gt_datas.size(0);
    int dt_size = dt_datas.size(0);
    int dc_size = dc_bboxes.size(0);
    
    // assigned_detection = np.zeros(det_size, dtype=bool)
    std::vector<bool> assigned_detection(dt_size, false);
    
    // ignored_threshold = (dt_scores < thresh)
    std::vector<bool> ignored_threshold(dt_size, false);
    for(int i=0; i<dt_size; i++) {
        if (dt_datas_a[i][dt_datas.size(1)-1] < thresh) {
            ignored_threshold[i] = true;
        }
    }
    
    const int NO_DETECTION = -10000000;
    int tp = 0, fp = 0, fn = 0;
    double similarity = 0;
    
    std::vector<float> thresholds;
    std::vector<float> delta;
    
    // Pre-calculate candidates for each GT to avoid O(N*M) inside matching loop
    // But since overlaps is N*M, we iterate it.
    // Python logic:
    // for i in range(gt_size):
    //    valid_indices = where(overlaps[:, i] > min_overlap)
    //    sort candidates
    
    for(int i=0; i<gt_size; i++) {
        if (ignored_gt_a[i] == -1) continue;
        
        int det_idx = -1;
        int valid_detection = NO_DETECTION;
        
        // Build candidates for this GT
        std::vector<Candidate> candidates;
        for(int j=0; j<dt_size; j++) {
            if (overlaps_a[j][i] > min_overlap) {
                candidates.push_back({j, overlaps_a[j][i], (int)ignored_det_a[j]});
            }
        }
        
        // Sort: 1. ignored_det == 0 (False < True? No, 0 is valid. 1 is ignored. -1 is dontcare)
        // Python: normal_mask = (curr_ignored == 0). norm_order = argsort(-norm_overlaps)
        // So ignored_det == 0 comes first.
        // Then overlap descending.
        // ign_indices = (curr_ignored == 1). Sorted by index?? Python: sorted_ign = np.sort(ign_indices)
        // So:
        // Group 1: ignored_det == 0, sorted by overlap desc
        // Group 2: ignored_det == 1, sorted by index asc
        // Ignore Group 3: ignored_det == -1 (already filtered by loop check below?)
        // Python loop: if (ignored_det[j] == -1) continue;
        
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            bool a_valid = (a.ignored_det == 0);
            bool b_valid = (b.ignored_det == 0);
            if (a_valid != b_valid) return a_valid > b_valid; // Valid first
            
            if (a_valid) {
                return a.overlap > b.overlap; // Descending overlap
            } else {
                return a.index < b.index; // Ascending index
            }
        });
        
        for (const auto& cand : candidates) {
            int j = cand.index;
            if (ignored_det_a[j] == -1) continue;
            if (assigned_detection[j]) continue;
            if (ignored_threshold[j]) continue;
            det_idx = j;
            valid_detection = 1;
            break;
        }
        
        if (valid_detection == NO_DETECTION && ignored_gt_a[i] == 0) {
            fn += 1;
        } else if (valid_detection != NO_DETECTION && (ignored_gt_a[i] == 1 || ignored_det_a[det_idx] == 1)) {
            assigned_detection[det_idx] = true;
        } else if (valid_detection != NO_DETECTION) {
            tp += 1;
            thresholds.push_back(dt_datas_a[det_idx][dt_datas.size(1)-1]);
            if (compute_aos) {
                 float gt_alpha = gt_datas_a[i][4];
                 float dt_alpha = dt_datas_a[det_idx][4];
                 delta.push_back(gt_alpha - dt_alpha);
            }
            assigned_detection[det_idx] = true;
        }
    }
    
    // FP Calculation
    // valid_mask = (~assigned_detection) & (ignored_det != -1) & (ignored_det != 1) & (~ignored_threshold)
    // iterate over detections
    
    // Special handling for metric == 0 (bbox) and dontcares
    // If metric == 0, check overlap with dc_bboxes
    
    int fp_adjustment = 0;
    int fp_raw = 0;
    
    // If metric==0 and dc_bboxes exist, we need to check overlaps for FPs
    // We can do this in batch or one by one.
    // For simplicity and since FPs are usually limited, one by one is OK, or call the 2D IoU kernel?
    // Calling kernel from here is hard without Tensor info.
    // Let's implement simple IoU on CPU for FP filtering if needed.
    
    for(int j=0; j<dt_size; j++) {
        if (assigned_detection[j]) continue;
        if (ignored_det_a[j] == -1 || ignored_det_a[j] == 1) continue;
        if (ignored_threshold[j]) continue;
        
        fp_raw++;
        
        if (metric == 0 && dc_size > 0) {
             // Check overlap with any DC
             float dt_x1 = dt_datas_a[j][0];
             float dt_y1 = dt_datas_a[j][1];
             float dt_x2 = dt_datas_a[j][2];
             float dt_y2 = dt_datas_a[j][3];
             float dt_area = (dt_x2 - dt_x1) * (dt_y2 - dt_y1);
             
             bool is_dc = false;
             auto dc_a = dc_bboxes.accessor<float, 2>();
             
             for(int k=0; k<dc_size; k++) {
                 float dc_x1 = dc_a[k][0];
                 float dc_y1 = dc_a[k][1];
                 float dc_x2 = dc_a[k][2];
                 float dc_y2 = dc_a[k][3];
                 
                 float iw = std::min(dt_x2, dc_x2) - std::max(dt_x1, dc_x1);
                 float ih = std::min(dt_y2, dc_y2) - std::max(dt_y1, dc_y1);
                 
                 if (iw > 0 && ih > 0) {
                     float inter = iw * ih;
                     float iou_val = inter / ((dc_x2 - dc_x1) * (dc_y2 - dc_y1) + dt_area - inter);
                     // metric 0 uses criterion 0 (inter / area_dt)? 
                     // image_box_overlap(dt, dc, 0). 0 means overlap / area_dt (area1).
                     // So inter / dt_area
                     float ov = inter / dt_area;
                     if (ov > min_overlap) {
                         is_dc = true;
                         break;
                     }
                 }
             }
             if (is_dc) fp_adjustment++;
        }
    }
    
    fp = fp_raw - fp_adjustment;
    
    if (compute_aos) {
        for(float d : delta) {
            similarity += (1.0 + cos(d)) / 2.0;
        }
    }
    
    // Return tensors
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tp_t = torch::tensor({(float)tp}, opts);
    torch::Tensor fp_t = torch::tensor({(float)fp}, opts);
    torch::Tensor fn_t = torch::tensor({(float)fn}, opts);
    torch::Tensor sim_t = torch::tensor({(float)similarity}, opts);
    
    // thresholds info
    torch::Tensor thresh_t = torch::from_blob(thresholds.data(), {(long)thresholds.size()}, opts).clone();
    
    return {tp_t, fp_t, fn_t, sim_t, thresh_t};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("d3_box_overlap_gpu", &iou3d_gpu, "3D Box Overlap / IoU (CUDA)");
    m.def("rotate_iou_gpu", &iou_bev_gpu, "Rotated Box IoU (CUDA)");
    m.def("iou2d_gpu", &iou2d_gpu, "2D Box IoU (CUDA)");
    m.def("compute_statistics_cpp", &compute_statistics_cpp, "Compute KITTI Statistics (CPU)");
}
