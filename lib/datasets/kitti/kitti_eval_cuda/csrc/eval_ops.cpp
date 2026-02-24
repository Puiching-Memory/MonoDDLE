/**
 * C++ implementation of all performance-critical eval functions:
 *   - image_box_overlap: 2D bbox IoU (float64)
 *   - d3_box_overlap_kernel: 3D box IoU (float32)
 *   - rotated IoU: float32 (done in CUDA kernel)
 *   - compute_statistics_jit: TP/FP/FN matching per image (was 51% of runtime)
 *   - fused_compute_statistics: batched accumulation with OpenMP (was 18% of runtime)
 *   - collect_thresholds: batched threshold collection with OpenMP
 *
 * OpenMP parallelizes across images within each batch (192 cores available).
 */
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstring>
#include <omp.h>

/**
 * 2D image bounding box overlap.
 * boxes:       [N, 4] (x1, y1, x2, y2) — preserves input dtype
 * query_boxes: [K, 4]
 * criterion: -1=IoU, 0=overlap/area(box), 1=overlap/area(query)
 * Returns: [N, K] same dtype as input
 */
torch::Tensor image_box_overlap_cpu(torch::Tensor boxes, torch::Tensor query_boxes, int criterion) {
    // Use double (float64) to match numpy default
    auto boxes_d = boxes.contiguous().to(torch::kFloat64);
    auto qboxes_d = query_boxes.contiguous().to(torch::kFloat64);
    auto boxes_a = boxes_d.accessor<double, 2>();
    auto qboxes_a = qboxes_d.accessor<double, 2>();
    int N = boxes_d.size(0);
    int K = qboxes_d.size(0);
    auto overlaps = torch::zeros({N, K}, torch::kFloat64);
    auto overlaps_a = overlaps.accessor<double, 2>();

    for (int k = 0; k < K; k++) {
        double qbox_area = (qboxes_a[k][2] - qboxes_a[k][0]) * (qboxes_a[k][3] - qboxes_a[k][1]);
        for (int n = 0; n < N; n++) {
            double iw = std::min(boxes_a[n][2], qboxes_a[k][2]) - std::max(boxes_a[n][0], qboxes_a[k][0]);
            if (iw > 0) {
                double ih = std::min(boxes_a[n][3], qboxes_a[k][3]) - std::max(boxes_a[n][1], qboxes_a[k][1]);
                if (ih > 0) {
                    double ua;
                    if (criterion == -1) {
                        ua = (boxes_a[n][2] - boxes_a[n][0]) * (boxes_a[n][3] - boxes_a[n][1]) + qbox_area - iw * ih;
                    } else if (criterion == 0) {
                        ua = (boxes_a[n][2] - boxes_a[n][0]) * (boxes_a[n][3] - boxes_a[n][1]);
                    } else if (criterion == 1) {
                        ua = qbox_area;
                    } else {
                        ua = 1.0;
                    }
                    overlaps_a[n][k] = iw * ih / ua;
                }
            }
        }
    }
    return overlaps;
}

/**
 * 3D box overlap kernel (applied on top of BEV rotated IoU).
 * Uses float32 throughout to match original behavior where:
 *   - rinc comes from rotate_iou_gpu_eval which outputs float32
 *   - boxes are originally float64 but the original code processes element by element
 *
 * boxes:  [N, 7] (x, y, z, w, h, l, ry) — camera coords
 * qboxes: [K, 7]
 * rinc:   [N, K] — pre-computed BEV rotated intersection area (float32)
 * Returns: [N, K] 3D IoU (float32)
 */
torch::Tensor d3_box_overlap_kernel_cpu(
    torch::Tensor boxes, torch::Tensor qboxes, torch::Tensor rinc, int criterion) {
    auto boxes_f = boxes.contiguous().to(torch::kFloat32);
    auto qboxes_f = qboxes.contiguous().to(torch::kFloat32);
    auto rinc_f = rinc.contiguous().to(torch::kFloat32);
    auto boxes_a = boxes_f.accessor<float, 2>();
    auto qboxes_a = qboxes_f.accessor<float, 2>();
    int N = boxes_f.size(0);
    int K = qboxes_f.size(0);

    auto result = rinc_f.clone();
    auto result_a = result.accessor<float, 2>();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            if (result_a[i][j] > 0) {
                float iw = std::min(boxes_a[i][1], qboxes_a[j][1]) -
                           std::max(boxes_a[i][1] - boxes_a[i][4], qboxes_a[j][1] - qboxes_a[j][4]);
                if (iw > 0) {
                    float area1 = boxes_a[i][3] * boxes_a[i][4] * boxes_a[i][5];
                    float area2 = qboxes_a[j][3] * qboxes_a[j][4] * qboxes_a[j][5];
                    float inc = iw * result_a[i][j];
                    float ua;
                    if (criterion == -1) ua = area1 + area2 - inc;
                    else if (criterion == 0) ua = area1;
                    else if (criterion == 1) ua = area2;
                    else ua = inc;
                    result_a[i][j] = inc / ua;
                } else {
                    result_a[i][j] = 0.0f;
                }
            }
        }
    }
    return result;
}

// Forward declaration
torch::Tensor rotate_iou_gpu(torch::Tensor boxes, torch::Tensor query_boxes, int criterion);

torch::Tensor bev_box_overlap(torch::Tensor boxes, torch::Tensor qboxes, int criterion) {
    return rotate_iou_gpu(boxes, qboxes, criterion);
}

torch::Tensor d3_box_overlap(torch::Tensor boxes, torch::Tensor qboxes, int criterion) {
    // Extract BEV: [x, z, w, l, ry] indices [0, 2, 3, 5, 6]
    auto gt_bev = torch::stack({
        boxes.select(1, 0), boxes.select(1, 2),
        boxes.select(1, 3), boxes.select(1, 5),
        boxes.select(1, 6)
    }, 1).contiguous();
    auto dt_bev = torch::stack({
        qboxes.select(1, 0), qboxes.select(1, 2),
        qboxes.select(1, 3), qboxes.select(1, 5),
        qboxes.select(1, 6)
    }, 1).contiguous();

    // criterion=2 for intersection area only (matching original)
    auto rinc = rotate_iou_gpu(gt_bev, dt_bev, 2);
    return d3_box_overlap_kernel_cpu(boxes, qboxes, rinc, criterion);
}

// ======================================================================
// compute_statistics_jit — C++ reimplementation
// This was the #1 hotspot: 1.8M calls, 16.9s (51% of total runtime).
// Now runs as a single C++ call with zero Python overhead.
// ======================================================================

/**
 * Compute 2D image box overlap using raw double pointers (no torch overhead).
 * Used internally by compute_statistics_jit for DontCare box matching.
 * This avoids the 866K torch.from_numpy()/tensor.numpy() conversions.
 */
static void image_box_overlap_raw(
    const double* boxes, int N,
    const double* qboxes, int K,
    double* overlaps, int criterion) {
    for (int k = 0; k < K; k++) {
        double qbox_area = (qboxes[k*4+2] - qboxes[k*4+0]) * (qboxes[k*4+3] - qboxes[k*4+1]);
        for (int n = 0; n < N; n++) {
            double iw = std::min(boxes[n*4+2], qboxes[k*4+2]) - std::max(boxes[n*4+0], qboxes[k*4+0]);
            if (iw > 0) {
                double ih = std::min(boxes[n*4+3], qboxes[k*4+3]) - std::max(boxes[n*4+1], qboxes[k*4+1]);
                if (ih > 0) {
                    double ua;
                    if (criterion == -1) {
                        ua = (boxes[n*4+2] - boxes[n*4+0]) * (boxes[n*4+3] - boxes[n*4+1]) + qbox_area - iw * ih;
                    } else if (criterion == 0) {
                        ua = (boxes[n*4+2] - boxes[n*4+0]) * (boxes[n*4+3] - boxes[n*4+1]);
                    } else if (criterion == 1) {
                        ua = qbox_area;
                    } else {
                        ua = 1.0;
                    }
                    overlaps[n * K + k] = iw * ih / ua;
                }
            }
        }
    }
}

/**
 * C++ compute_statistics_jit — exact match of Python implementation.
 * Returns tuple: (tp, fp, fn, similarity, thresholds_tensor)
 */
std::tuple<int64_t, int64_t, int64_t, double, torch::Tensor>
compute_statistics_jit_cpp(
    torch::Tensor overlaps_t,    // [det_size, gt_size] float64
    torch::Tensor gt_datas_t,    // [gt_size, 5] float64 (bbox4 + alpha)
    torch::Tensor dt_datas_t,    // [det_size, 6] float64 (bbox4 + alpha + score)
    torch::Tensor ignored_gt_t,  // [gt_size] int64
    torch::Tensor ignored_det_t, // [det_size] int64
    torch::Tensor dc_bboxes_t,   // [num_dc, 4] float64
    int64_t metric,
    double min_overlap,
    double thresh,
    bool compute_fp,
    bool compute_aos) {

    auto overlaps = overlaps_t.contiguous().to(torch::kFloat64);
    auto gt_datas = gt_datas_t.contiguous().to(torch::kFloat64);
    auto dt_datas = dt_datas_t.contiguous().to(torch::kFloat64);
    auto ignored_gt = ignored_gt_t.contiguous().to(torch::kInt64);
    auto ignored_det = ignored_det_t.contiguous().to(torch::kInt64);
    auto dc_bboxes = dc_bboxes_t.contiguous().to(torch::kFloat64);

    int det_size = dt_datas.size(0);
    int gt_size = gt_datas.size(0);
    int num_dc = dc_bboxes.size(0);

    auto ov_a = overlaps.accessor<double, 2>();
    auto gt_a = gt_datas.accessor<double, 2>();
    auto dt_a = dt_datas.accessor<double, 2>();
    auto ign_gt_a = ignored_gt.accessor<int64_t, 1>();
    auto ign_det_a = ignored_det.accessor<int64_t, 1>();

    // dt_scores = dt_datas[:, -1], dt_alphas = dt_datas[:, 4], gt_alphas = gt_datas[:, 4]
    std::vector<bool> assigned_detection(det_size, false);
    std::vector<bool> ignored_threshold(det_size, false);

    if (compute_fp) {
        for (int i = 0; i < det_size; i++) {
            if (dt_a[i][dt_datas.size(1)-1] < thresh) {
                ignored_threshold[i] = true;
            }
        }
    }

    const double NO_DETECTION = -10000000.0;
    int64_t tp = 0, fp = 0, fn = 0;
    double similarity = 0.0;
    std::vector<double> thresholds_vec;
    thresholds_vec.reserve(gt_size);
    std::vector<double> delta_vec;
    delta_vec.reserve(gt_size);

    for (int i = 0; i < gt_size; i++) {
        if (ign_gt_a[i] == -1) continue;
        int det_idx = -1;
        double valid_detection = NO_DETECTION;
        double max_overlap = 0.0;
        bool assigned_ignored_det = false;

        for (int j = 0; j < det_size; j++) {
            if (ign_det_a[j] == -1) continue;
            if (assigned_detection[j]) continue;
            if (ignored_threshold[j]) continue;
            double overlap = ov_a[j][i];
            double dt_score = dt_a[j][dt_datas.size(1)-1];
            if (!compute_fp && overlap > min_overlap && dt_score > valid_detection) {
                det_idx = j;
                valid_detection = dt_score;
            } else if (compute_fp && overlap > min_overlap
                       && (overlap > max_overlap || assigned_ignored_det)
                       && ign_det_a[j] == 0) {
                max_overlap = overlap;
                det_idx = j;
                valid_detection = 1.0;
                assigned_ignored_det = false;
            } else if (compute_fp && overlap > min_overlap
                       && valid_detection == NO_DETECTION
                       && ign_det_a[j] == 1) {
                det_idx = j;
                valid_detection = 1.0;
                assigned_ignored_det = true;
            }
        }

        if (valid_detection == NO_DETECTION && ign_gt_a[i] == 0) {
            fn += 1;
        } else if (valid_detection != NO_DETECTION
                   && (ign_gt_a[i] == 1 || ign_det_a[det_idx] == 1)) {
            assigned_detection[det_idx] = true;
        } else if (valid_detection != NO_DETECTION) {
            tp += 1;
            thresholds_vec.push_back(dt_a[det_idx][dt_datas.size(1)-1]);
            if (compute_aos) {
                delta_vec.push_back(gt_a[i][4] - dt_a[det_idx][4]);
            }
            assigned_detection[det_idx] = true;
        }
    }

    if (compute_fp) {
        for (int i = 0; i < det_size; i++) {
            if (!(assigned_detection[i] || ign_det_a[i] == -1
                  || ign_det_a[i] == 1 || ignored_threshold[i])) {
                fp += 1;
            }
        }
        int64_t nstuff = 0;
        if (metric == 0 && num_dc > 0) {
            // Compute overlaps_dt_dc inline (avoiding torch overhead)
            std::vector<double> overlaps_dt_dc(det_size * num_dc, 0.0);
            auto dc_a = dc_bboxes.accessor<double, 2>();
            // dt_bboxes = dt_datas[:, :4]
            std::vector<double> dt_bboxes(det_size * 4);
            for (int j = 0; j < det_size; j++) {
                dt_bboxes[j*4+0] = dt_a[j][0];
                dt_bboxes[j*4+1] = dt_a[j][1];
                dt_bboxes[j*4+2] = dt_a[j][2];
                dt_bboxes[j*4+3] = dt_a[j][3];
            }
            std::vector<double> dc_boxes(num_dc * 4);
            for (int k = 0; k < num_dc; k++) {
                dc_boxes[k*4+0] = dc_a[k][0];
                dc_boxes[k*4+1] = dc_a[k][1];
                dc_boxes[k*4+2] = dc_a[k][2];
                dc_boxes[k*4+3] = dc_a[k][3];
            }
            image_box_overlap_raw(dt_bboxes.data(), det_size,
                                  dc_boxes.data(), num_dc,
                                  overlaps_dt_dc.data(), 0);
            for (int ii = 0; ii < num_dc; ii++) {
                for (int j = 0; j < det_size; j++) {
                    if (assigned_detection[j]) continue;
                    if (ign_det_a[j] == -1 || ign_det_a[j] == 1) continue;
                    if (ignored_threshold[j]) continue;
                    if (overlaps_dt_dc[j * num_dc + ii] > min_overlap) {
                        assigned_detection[j] = true;
                        nstuff += 1;
                    }
                }
            }
        }
        fp -= nstuff;
        if (compute_aos) {
            int delta_idx = (int)delta_vec.size();
            // tmp = zeros(fp + delta_idx)
            // tmp[fp:] = (1 + cos(delta)) / 2
            if (tp > 0 || fp > 0) {
                double sum_sim = 0.0;
                for (int i = 0; i < delta_idx; i++) {
                    sum_sim += (1.0 + std::cos(delta_vec[i])) / 2.0;
                }
                similarity = sum_sim;
            } else {
                similarity = -1.0;
            }
        }
    }

    auto thresholds_out = torch::zeros({(int64_t)thresholds_vec.size()}, torch::kFloat64);
    if (!thresholds_vec.empty()) {
        auto th_a = thresholds_out.accessor<double, 1>();
        for (size_t i = 0; i < thresholds_vec.size(); i++) {
            th_a[i] = thresholds_vec[i];
        }
    }
    return std::make_tuple(tp, fp, fn, similarity, thresholds_out);
}

// ======================================================================
// fused_compute_statistics — OpenMP-parallel C++ reimplementation
// Parallelizes across thresholds (each thread accumulates independent tp/fp/fn/sim,
// then reduces into pr). Inner loop over images is sequential per threshold
// since compute_statistics_jit has serial dependencies (assigned_detection).
// ======================================================================

void fused_compute_statistics_cpp(
    torch::Tensor overlaps_t,     // [total_dt, total_gt] float64
    torch::Tensor pr_t,           // [num_thresholds, 4] float64 (tp, fp, fn, similarity) — modified in-place
    torch::Tensor gt_nums_t,      // [num_images] int64
    torch::Tensor dt_nums_t,      // [num_images] int64
    torch::Tensor dc_nums_t,      // [num_images] int64
    torch::Tensor gt_datas_t,     // [total_gt, 5] float64
    torch::Tensor dt_datas_t,     // [total_dt, 6] float64
    torch::Tensor dontcares_t,    // [total_dc, 4] float64
    torch::Tensor ignored_gts_t,  // [total_gt] int64
    torch::Tensor ignored_dets_t, // [total_dt] int64
    int64_t metric,
    double min_overlap,
    torch::Tensor thresholds_t,   // [num_thresholds] float64
    bool compute_aos) {

    auto overlaps = overlaps_t.contiguous().to(torch::kFloat64);
    auto pr = pr_t.contiguous();  // Must be float64, in-place
    auto gt_nums = gt_nums_t.contiguous().to(torch::kInt64);
    auto dt_nums = dt_nums_t.contiguous().to(torch::kInt64);
    auto dc_nums = dc_nums_t.contiguous().to(torch::kInt64);
    auto gt_datas = gt_datas_t.contiguous().to(torch::kFloat64);
    auto dt_datas = dt_datas_t.contiguous().to(torch::kFloat64);
    auto dontcares = dontcares_t.contiguous().to(torch::kFloat64);
    auto ignored_gts = ignored_gts_t.contiguous().to(torch::kInt64);
    auto ignored_dets = ignored_dets_t.contiguous().to(torch::kInt64);
    auto thresholds = thresholds_t.contiguous().to(torch::kFloat64);

    int num_images = gt_nums.size(0);
    int num_thresholds = thresholds.size(0);

    auto pr_a = pr.accessor<double, 2>();
    auto gt_nums_a = gt_nums.accessor<int64_t, 1>();
    auto dt_nums_a = dt_nums.accessor<int64_t, 1>();
    auto dc_nums_a = dc_nums.accessor<int64_t, 1>();
    auto thresholds_a = thresholds.accessor<double, 1>();

    // Precompute image start offsets for O(1) slicing
    std::vector<int64_t> gt_offsets(num_images + 1, 0);
    std::vector<int64_t> dt_offsets(num_images + 1, 0);
    std::vector<int64_t> dc_offsets(num_images + 1, 0);
    for (int i = 0; i < num_images; i++) {
        gt_offsets[i + 1] = gt_offsets[i] + gt_nums_a[i];
        dt_offsets[i + 1] = dt_offsets[i] + dt_nums_a[i];
        dc_offsets[i + 1] = dc_offsets[i] + dc_nums_a[i];
    }

    // Parallelize across thresholds — each threshold is independent
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < num_thresholds; t++) {
        double thresh = thresholds_a[t];
        int64_t total_tp = 0, total_fp = 0, total_fn = 0;
        double total_sim = 0.0;
        bool has_sim = false;

        for (int i = 0; i < num_images; i++) {
            auto overlap_i = overlaps.slice(0, dt_offsets[i], dt_offsets[i + 1])
                                     .slice(1, gt_offsets[i], gt_offsets[i + 1]);
            auto gt_data_i = gt_datas.slice(0, gt_offsets[i], gt_offsets[i + 1]);
            auto dt_data_i = dt_datas.slice(0, dt_offsets[i], dt_offsets[i + 1]);
            auto ignored_gt_i = ignored_gts.slice(0, gt_offsets[i], gt_offsets[i + 1]);
            auto ignored_det_i = ignored_dets.slice(0, dt_offsets[i], dt_offsets[i + 1]);
            auto dontcare_i = dontcares.slice(0, dc_offsets[i], dc_offsets[i + 1]);

            auto [tp, fp, fn, sim, _] = compute_statistics_jit_cpp(
                overlap_i, gt_data_i, dt_data_i,
                ignored_gt_i, ignored_det_i, dontcare_i,
                metric, min_overlap, thresh,
                /*compute_fp=*/true, /*compute_aos=*/compute_aos);
            total_tp += tp;
            total_fp += fp;
            total_fn += fn;
            if (sim != -1.0) {
                total_sim += sim;
                has_sim = true;
            }
        }

        // Thread-safe update (each t is unique, no race)
        pr_a[t][0] += total_tp;
        pr_a[t][1] += total_fp;
        pr_a[t][2] += total_fn;
        if (has_sim) {
            pr_a[t][3] += total_sim;
        }
    }
}

// ======================================================================
// collect_thresholds — batched + parallel threshold collection
// Replaces the Python per-image loop calling compute_statistics_jit(..., compute_fp=False)
// which was 67,842 calls taking ~0.85s.
// ======================================================================

torch::Tensor collect_thresholds_cpp(
    std::vector<torch::Tensor> overlaps_list,   // list of [det_i, gt_i] per image
    std::vector<torch::Tensor> gt_datas_list,   // list of [gt_i, 5] per image
    std::vector<torch::Tensor> dt_datas_list,   // list of [det_i, 6] per image
    std::vector<torch::Tensor> ignored_gts_list, // list of [gt_i] per image
    std::vector<torch::Tensor> ignored_dets_list, // list of [det_i] per image
    std::vector<torch::Tensor> dontcares_list,  // list of [dc_i, 4] per image
    int64_t metric,
    double min_overlap) {

    int num_images = (int)overlaps_list.size();

    // Parallel collection: each image produces a vector of thresholds
    std::vector<std::vector<double>> per_image_thresholds(num_images);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_images; i++) {
        auto [tp, fp, fn, sim, thresholds_out] = compute_statistics_jit_cpp(
            overlaps_list[i], gt_datas_list[i], dt_datas_list[i],
            ignored_gts_list[i], ignored_dets_list[i], dontcares_list[i],
            metric, min_overlap, 0.0,
            /*compute_fp=*/false, /*compute_aos=*/false);
        int n = thresholds_out.size(0);
        if (n > 0) {
            auto th_a = thresholds_out.accessor<double, 1>();
            per_image_thresholds[i].resize(n);
            for (int j = 0; j < n; j++) {
                per_image_thresholds[i][j] = th_a[j];
            }
        }
    }

    // Flatten
    int total = 0;
    for (auto& v : per_image_thresholds) total += (int)v.size();
    auto result = torch::zeros({total}, torch::kFloat64);
    if (total > 0) {
        auto r_a = result.accessor<double, 1>();
        int idx = 0;
        for (auto& v : per_image_thresholds) {
            for (double d : v) {
                r_a[idx++] = d;
            }
        }
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotate_iou_gpu", &rotate_iou_gpu, "Rotated IoU GPU (float32)",
          py::arg("boxes"), py::arg("query_boxes"), py::arg("criterion") = -1);
    m.def("image_box_overlap", &image_box_overlap_cpu, "2D Image Box Overlap (float64)",
          py::arg("boxes"), py::arg("query_boxes"), py::arg("criterion") = -1);
    m.def("d3_box_overlap_kernel", &d3_box_overlap_kernel_cpu, "3D Box Overlap Kernel",
          py::arg("boxes"), py::arg("qboxes"), py::arg("rinc"), py::arg("criterion") = -1);
    m.def("bev_box_overlap", &bev_box_overlap, "BEV Box Overlap (float32)",
          py::arg("boxes"), py::arg("qboxes"), py::arg("criterion") = -1);
    m.def("d3_box_overlap", &d3_box_overlap, "3D Box Overlap",
          py::arg("boxes"), py::arg("qboxes"), py::arg("criterion") = -1);
    m.def("fused_compute_statistics", &fused_compute_statistics_cpp,
          "Fused TP/FP/FN accumulation (C++)",
          py::arg("overlaps"), py::arg("pr"),
          py::arg("gt_nums"), py::arg("dt_nums"), py::arg("dc_nums"),
          py::arg("gt_datas"), py::arg("dt_datas"), py::arg("dontcares"),
          py::arg("ignored_gts"), py::arg("ignored_dets"),
          py::arg("metric"), py::arg("min_overlap"),
          py::arg("thresholds"), py::arg("compute_aos"));
    m.def("compute_statistics_jit", &compute_statistics_jit_cpp,
          "Compute TP/FP/FN statistics (C++)",
          py::arg("overlaps"), py::arg("gt_datas"), py::arg("dt_datas"),
          py::arg("ignored_gt"), py::arg("ignored_det"),
          py::arg("dc_bboxes"), py::arg("metric"), py::arg("min_overlap"),
          py::arg("thresh"), py::arg("compute_fp"), py::arg("compute_aos"));
    m.def("collect_thresholds", &collect_thresholds_cpp,
          "Batched threshold collection with OpenMP (C++)",
          py::arg("overlaps_list"), py::arg("gt_datas_list"), py::arg("dt_datas_list"),
          py::arg("ignored_gts_list"), py::arg("ignored_dets_list"),
          py::arg("dontcares_list"), py::arg("metric"), py::arg("min_overlap"));
}
