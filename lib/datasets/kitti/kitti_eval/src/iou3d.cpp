/*
 * KITTI 评估 C++ 绑定层 (优化版)
 * ================================
 *
 * 提供 PyTorch C++ 扩展接口：
 *   - 2D/BEV/3D IoU GPU 计算
 *   - compute_statistics_internal: 零拷贝、无 Tensor 创建的内部评估逻辑
 *   - batch_collect_thresholds: 批量收集所有图片的 TP 阈值 (第一阶段)
 *   - batch_compute_pr: 批量计算所有阈值的 PR (第二阶段)
 *
 * 性能优化:
 *   - 内部函数使用原始指针，消除 per-image Tensor 创建开销
 *   - 批量 API 将整个 Python 循环合并到单次 C++ 调用
 *   - DontCare 2D IoU 内联计算，无额外 kernel 调用
 *
 * 编译: torch.utils.cpp_extension.load() JIT
 */

#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// ═══════════════════ CUDA 内核前向声明 ═══════════════════

void launch_rotate_iou_kernel(
    int64_t N, int64_t K,
    const float* boxes, const float* query_boxes,
    float* iou, int criterion);

void launch_iou2d_kernel(
    int N, int K,
    const float* boxes, const float* query_boxes,
    float* out, int criterion);

void launch_iou3d_kernel(
    int N, int K,
    const float* boxes, const float* query_boxes,
    float* out, int criterion);

// ═══════════════════ 输入检查宏 ═══════════════════

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ═══════════════════ IoU 接口 ═══════════════════

at::Tensor iou2d_gpu(at::Tensor boxes, at::Tensor query_boxes, int criterion) {
    CHECK_INPUT(boxes);
    CHECK_INPUT(query_boxes);
    int N = boxes.size(0);
    int K = query_boxes.size(0);
    auto out = torch::zeros({N, K}, boxes.options());
    if (N > 0 && K > 0) {
        launch_iou2d_kernel(N, K,
            boxes.data_ptr<float>(),
            query_boxes.data_ptr<float>(),
            out.data_ptr<float>(), criterion);
    }
    return out;
}

at::Tensor rotate_iou_gpu(at::Tensor boxes, at::Tensor query_boxes, int criterion) {
    CHECK_INPUT(boxes);
    CHECK_INPUT(query_boxes);
    int64_t N = boxes.size(0);
    int64_t K = query_boxes.size(0);
    auto out = torch::zeros({N, K}, boxes.options());
    if (N > 0 && K > 0) {
        launch_rotate_iou_kernel(N, K,
            boxes.data_ptr<float>(),
            query_boxes.data_ptr<float>(),
            out.data_ptr<float>(), criterion);
    }
    return out;
}

at::Tensor d3_box_overlap_gpu(at::Tensor boxes, at::Tensor query_boxes, int criterion) {
    CHECK_INPUT(boxes);
    CHECK_INPUT(query_boxes);
    int N = boxes.size(0);
    int K = query_boxes.size(0);
    auto out = torch::zeros({N, K}, boxes.options());
    if (N > 0 && K > 0) {
        launch_iou3d_kernel(N, K,
            boxes.data_ptr<float>(),
            query_boxes.data_ptr<float>(),
            out.data_ptr<float>(), criterion);
    }
    return out;
}

// ═══════════════════ 内部评估逻辑 (零 Tensor 开销) ═══════════════════

struct StatResult {
    int tp = 0;
    int fp = 0;
    int fn = 0;
    double similarity = 0.0;
    std::vector<float> thresholds;
};

/*
 * 核心评估统计量计算 — 纯指针操作，无 Tensor 创建
 *
 * 精确复刻 kitti_eval_python/eval.py::compute_statistics_jit
 */
static inline StatResult compute_statistics_internal(
    const float*   overlaps,      // (det_size, gt_size) row-major
    const float*   gt_datas,      // (gt_size, 5) [bbox(4), alpha]
    const float*   dt_datas,      // (det_size, dt_cols) [bbox(4), alpha, score]
    const int64_t* ignored_gt,    // (gt_size,)
    const int64_t* ignored_det,   // (det_size,)
    const float*   dc_bboxes,     // (dc_size, 4) — 可为 nullptr 当 dc_size==0
    int det_size, int gt_size, int dc_size, int dt_cols,
    int metric, double min_overlap, double thresh,
    bool compute_fp, bool compute_aos)
{
    StatResult result;
    // dt_scores = dt_datas[:, dt_cols-1]

    std::vector<bool> assigned(det_size, false);
    std::vector<bool> ignored_threshold(det_size, false);

    if (compute_fp) {
        for (int i = 0; i < det_size; i++) {
            if (dt_datas[i * dt_cols + dt_cols - 1] < (float)thresh)
                ignored_threshold[i] = true;
        }
    }

    constexpr int NO_DETECTION = -10000000;
    std::vector<float> delta_vec;

    for (int i = 0; i < gt_size; i++) {
        if (ignored_gt[i] == -1) continue;

        int det_idx = -1;
        float valid_detection = (float)NO_DETECTION;
        float max_overlap = 0.0f;
        bool assigned_ignored_det = false;

        for (int j = 0; j < det_size; j++) {
            if (ignored_det[j] == -1 || assigned[j] || ignored_threshold[j])
                continue;

            float overlap = overlaps[j * gt_size + i];
            float dt_score = dt_datas[j * dt_cols + dt_cols - 1];

            if (!compute_fp && overlap > (float)min_overlap
                && dt_score > valid_detection) {
                det_idx = j;
                valid_detection = dt_score;
            } else if (compute_fp && overlap > (float)min_overlap
                       && (overlap > max_overlap || assigned_ignored_det)
                       && ignored_det[j] == 0) {
                max_overlap = overlap;
                det_idx = j;
                valid_detection = 1.0f;
                assigned_ignored_det = false;
            } else if (compute_fp && overlap > (float)min_overlap
                       && valid_detection == (float)NO_DETECTION
                       && ignored_det[j] == 1) {
                det_idx = j;
                valid_detection = 1.0f;
                assigned_ignored_det = true;
            }
        }

        if (valid_detection == (float)NO_DETECTION && ignored_gt[i] == 0) {
            result.fn++;
        } else if (valid_detection != (float)NO_DETECTION
                   && (ignored_gt[i] == 1 || ignored_det[det_idx] == 1)) {
            assigned[det_idx] = true;
        } else if (valid_detection != (float)NO_DETECTION) {
            result.tp++;
            result.thresholds.push_back(dt_datas[det_idx * dt_cols + dt_cols - 1]);
            if (compute_aos) {
                float gt_alpha = gt_datas[i * 5 + 4];
                float dt_alpha = dt_datas[det_idx * dt_cols + 4];
                delta_vec.push_back(gt_alpha - dt_alpha);
            }
            assigned[det_idx] = true;
        }
    }

    if (compute_fp) {
        for (int i = 0; i < det_size; i++) {
            if (!(assigned[i] || ignored_det[i] == -1
                  || ignored_det[i] == 1 || ignored_threshold[i]))
                result.fp++;
        }

        // DontCare FP 扣除 (仅 metric==0)
        int nstuff = 0;
        if (metric == 0 && dc_size > 0 && dc_bboxes != nullptr) {
            for (int dc_i = 0; dc_i < dc_size; dc_i++) {
                for (int j = 0; j < det_size; j++) {
                    if (assigned[j] || ignored_det[j] == -1
                        || ignored_det[j] == 1 || ignored_threshold[j])
                        continue;

                    float dt_x1 = dt_datas[j * dt_cols + 0];
                    float dt_y1 = dt_datas[j * dt_cols + 1];
                    float dt_x2 = dt_datas[j * dt_cols + 2];
                    float dt_y2 = dt_datas[j * dt_cols + 3];
                    float dc_x1 = dc_bboxes[dc_i * 4 + 0];
                    float dc_y1 = dc_bboxes[dc_i * 4 + 1];
                    float dc_x2 = dc_bboxes[dc_i * 4 + 2];
                    float dc_y2 = dc_bboxes[dc_i * 4 + 3];

                    float iw = std::min(dt_x2, dc_x2) - std::max(dt_x1, dc_x1);
                    if (iw <= 0) continue;
                    float ih = std::min(dt_y2, dc_y2) - std::max(dt_y1, dc_y1);
                    if (ih <= 0) continue;

                    float dt_area = (dt_x2 - dt_x1) * (dt_y2 - dt_y1);
                    float ov = (dt_area > 0) ? (iw * ih) / dt_area : 0.0f;

                    if (ov > (float)min_overlap) {
                        assigned[j] = true;
                        nstuff++;
                    }
                }
            }
        }
        result.fp -= nstuff;

        if (compute_aos) {
            if (result.tp > 0 || result.fp > 0) {
                result.similarity = 0.0;
                for (size_t k = 0; k < delta_vec.size(); k++)
                    result.similarity += (1.0 + cos((double)delta_vec[k])) / 2.0;
            } else {
                result.similarity = -1.0;
            }
        }
    }

    return result;
}

// ═══════════ Python 兼容的单图片接口 (保留) ═══════════

std::vector<torch::Tensor> compute_statistics_cpp(
    at::Tensor overlaps, at::Tensor gt_datas, at::Tensor dt_datas,
    at::Tensor ignored_gt, at::Tensor ignored_det, at::Tensor dc_bboxes,
    int metric, double min_overlap, double thresh,
    bool compute_fp, bool compute_aos)
{
    int gt_size = gt_datas.size(0);
    int det_size = dt_datas.size(0);
    int dc_size = dc_bboxes.size(0);
    int dt_cols = dt_datas.size(1);

    StatResult r = compute_statistics_internal(
        overlaps.data_ptr<float>(),
        gt_datas.data_ptr<float>(),
        dt_datas.data_ptr<float>(),
        ignored_gt.data_ptr<int64_t>(),
        ignored_det.data_ptr<int64_t>(),
        dc_size > 0 ? dc_bboxes.data_ptr<float>() : nullptr,
        det_size, gt_size, dc_size, dt_cols,
        metric, min_overlap, thresh, compute_fp, compute_aos);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor thresh_t;
    if (r.thresholds.empty()) {
        thresh_t = torch::zeros({0}, opts);
    } else {
        thresh_t = torch::from_blob(
            r.thresholds.data(), {(long)r.thresholds.size()}, opts
        ).clone();
    }

    return {
        torch::tensor({(float)r.tp}, opts),
        torch::tensor({(float)r.fp}, opts),
        torch::tensor({(float)r.fn}, opts),
        torch::tensor({(float)r.similarity}, opts),
        thresh_t,
    };
}

// ═══════════════════ 批量 API (核心性能优化) ═══════════════════

/*
 * batch_collect_thresholds — 第一阶段批量收集所有 TP 阈值
 *
 * 将 Python 侧的 per-image 循环完全移入 C++，消除:
 *   1) Python 循环开销
 *   2) 每次调用的 6× Tensor 创建/转换开销
 *   3) Python list.extend() 开销
 *
 * 参数:
 *   overlap:      (total_dt, total_gt) — 分区内的连续重叠矩阵
 *   gt_datas:     (total_gt, 5)
 *   dt_datas:     (total_dt, 6)
 *   ignored_gt:   (total_gt,)
 *   ignored_det:  (total_dt,)
 *   dc_bboxes:    (total_dc, 4)
 *   gt_nums / dt_nums / dc_nums: (num_images,) — 每张图片的数量
 *
 * 返回: 1D float Tensor，所有 TP 检测的置信分数
 */
torch::Tensor batch_collect_thresholds(
    at::Tensor overlap,
    at::Tensor gt_datas,
    at::Tensor dt_datas,
    at::Tensor ignored_gt,
    at::Tensor ignored_det,
    at::Tensor dc_bboxes,
    at::Tensor gt_nums,
    at::Tensor dt_nums,
    at::Tensor dc_nums,
    int metric,
    double min_overlap)
{
    int num_images = gt_nums.size(0);
    auto gn = gt_nums.accessor<int64_t, 1>();
    auto dn = dt_nums.accessor<int64_t, 1>();
    auto dcn = dc_nums.accessor<int64_t, 1>();

    int total_gt = gt_datas.size(0);
    int dt_cols = dt_datas.size(1);
    const float*   ov_ptr = overlap.data_ptr<float>();
    const float*   gt_ptr = gt_datas.data_ptr<float>();
    const float*   dt_ptr = dt_datas.data_ptr<float>();
    const int64_t* igt_ptr = ignored_gt.data_ptr<int64_t>();
    const int64_t* idt_ptr = ignored_det.data_ptr<int64_t>();
    const float*   dc_ptr = dc_bboxes.size(0) > 0 ? dc_bboxes.data_ptr<float>() : nullptr;

    std::vector<float> all_thresholds;
    all_thresholds.reserve(total_gt);  // 上界预分配

    // 预提取每张图片的偏移量 (用于 OpenMP 并行)
    struct ImgOffset { int64_t gt_off, dt_off, dc_off, g, d, dc; };
    std::vector<ImgOffset> offsets(num_images);
    {
        int64_t gt_off = 0, dt_off = 0, dc_off = 0;
        for (int img = 0; img < num_images; img++) {
            offsets[img] = {gt_off, dt_off, dc_off, gn[img], dn[img], dcn[img]};
            gt_off += gn[img]; dt_off += dn[img]; dc_off += dcn[img];
        }
    }

    // OpenMP 并行: 每个线程收集自己的 thresholds, 最后合并
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif
    std::vector<std::vector<float>> tl_thresholds(num_threads);

#pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        tl_thresholds[tid].reserve(total_gt / num_threads + 64);

#pragma omp for schedule(dynamic, 4)
        for (int img = 0; img < num_images; img++) {
            auto& off = offsets[img];
            int64_t g = off.g, d = off.d, dc = off.dc;

            std::vector<float> sub_ov(d * g);
            for (int64_t r = 0; r < d; r++) {
                std::memcpy(&sub_ov[r * g],
                            ov_ptr + (off.dt_off + r) * total_gt + off.gt_off,
                            g * sizeof(float));
            }

            StatResult res = compute_statistics_internal(
                sub_ov.data(),
                gt_ptr + off.gt_off * 5,
                dt_ptr + off.dt_off * dt_cols,
                igt_ptr + off.gt_off,
                idt_ptr + off.dt_off,
                (dc > 0 && dc_ptr) ? dc_ptr + off.dc_off * 4 : nullptr,
                (int)d, (int)g, (int)dc, dt_cols,
                metric, min_overlap, 0.0, false, false);

            tl_thresholds[tid].insert(tl_thresholds[tid].end(),
                                      res.thresholds.begin(), res.thresholds.end());
        }
    }

    // 合并所有线程的 thresholds
    for (int tid = 0; tid < num_threads; tid++) {
        all_thresholds.insert(all_thresholds.end(),
                              tl_thresholds[tid].begin(), tl_thresholds[tid].end());
    }

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    if (all_thresholds.empty())
        return torch::zeros({0}, opts);
    return torch::from_blob(
        all_thresholds.data(), {(long)all_thresholds.size()}, opts
    ).clone();
}

/*
 * batch_compute_pr — 第二阶段批量计算 PR
 *
 * 替代 Python 侧的 fused_compute_statistics，将
 * num_thresholds × num_images 的双重循环完全合并到 C++。
 *
 * 返回: (num_thresholds, 4) float64 Tensor [tp, fp, fn, similarity]
 */
at::Tensor batch_compute_pr(
    at::Tensor overlap,
    at::Tensor gt_datas,
    at::Tensor dt_datas,
    at::Tensor ignored_gt,
    at::Tensor ignored_det,
    at::Tensor dc_bboxes,
    at::Tensor gt_nums,
    at::Tensor dt_nums,
    at::Tensor dc_nums,
    int metric,
    double min_overlap,
    at::Tensor thresholds,
    bool compute_aos)
{
    int num_images = gt_nums.size(0);
    int num_thresholds = thresholds.size(0);
    auto gn = gt_nums.accessor<int64_t, 1>();
    auto dn = dt_nums.accessor<int64_t, 1>();
    auto dcn = dc_nums.accessor<int64_t, 1>();
    const float* thresh_ptr = thresholds.data_ptr<float>();

    int total_gt = gt_datas.size(0);
    int dt_cols = dt_datas.size(1);
    const float*   ov_ptr = overlap.data_ptr<float>();
    const float*   gt_ptr = gt_datas.data_ptr<float>();
    const float*   dt_ptr = dt_datas.data_ptr<float>();
    const int64_t* igt_ptr = ignored_gt.data_ptr<int64_t>();
    const int64_t* idt_ptr = ignored_det.data_ptr<int64_t>();
    const float*   dc_ptr = dc_bboxes.size(0) > 0 ? dc_bboxes.data_ptr<float>() : nullptr;

    // 输出: (num_thresholds, 4) double [tp, fp, fn, similarity]
    auto pr = torch::zeros({num_thresholds, 4}, torch::kFloat64);
    auto pr_ptr = pr.data_ptr<double>();

    // 预提取所有子矩阵 (一次性完成，避免在阈值循环中重复)
    struct ImageData {
        std::vector<float> ov;
        int det_size, gt_size, dc_size;
        int64_t gt_off, dt_off, dc_off;
    };
    std::vector<ImageData> images(num_images);

    int64_t gt_off = 0, dt_off = 0, dc_off = 0;
    for (int img = 0; img < num_images; img++) {
        int64_t g = gn[img], d = dn[img], dc = dcn[img];
        images[img].det_size = (int)d;
        images[img].gt_size = (int)g;
        images[img].dc_size = (int)dc;
        images[img].gt_off = gt_off;
        images[img].dt_off = dt_off;
        images[img].dc_off = dc_off;

        images[img].ov.resize(d * g);
        for (int64_t r = 0; r < d; r++) {
            std::memcpy(&images[img].ov[r * g],
                        ov_ptr + (dt_off + r) * total_gt + gt_off,
                        g * sizeof(float));
        }

        gt_off += g;
        dt_off += d;
        dc_off += dc;
    }

    // OpenMP 并行: 每个线程处理一部分图片, 使用线程局部 PR 累加器
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    // 线程局部 PR: (num_threads, num_thresholds, 4)
    std::vector<double> tl_pr(num_threads * num_thresholds * 4, 0.0);

#pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        double* my_pr = tl_pr.data() + tid * num_thresholds * 4;

        // 每个线程自己的缓冲区
        int max_det = 0, max_gt = 0;
        for (int img = 0; img < num_images; img++) {
            max_det = std::max(max_det, images[img].det_size);
            max_gt  = std::max(max_gt,  images[img].gt_size);
        }
        std::vector<bool> assigned(max_det, false);
        std::vector<bool> ignored_threshold(max_det, false);
        std::vector<float> delta_vec;
        delta_vec.reserve(max_gt);

#pragma omp for schedule(dynamic, 4)
        for (int img = 0; img < num_images; img++) {
            auto& im = images[img];
            int det_size = im.det_size;
            int gt_size  = im.gt_size;
            int dc_size  = im.dc_size;

            const float*   img_ov  = im.ov.data();
            const float*   img_gt  = gt_ptr + im.gt_off * 5;
            const float*   img_dt  = dt_ptr + im.dt_off * dt_cols;
            const int64_t* img_igt = igt_ptr + im.gt_off;
            const int64_t* img_idt = idt_ptr + im.dt_off;
            const float*   img_dc  = (dc_size > 0 && dc_ptr) ? dc_ptr + im.dc_off * 4 : nullptr;

            if (det_size == 0 && gt_size == 0) continue;

            for (int t = 0; t < num_thresholds; t++) {
                float thresh = thresh_ptr[t];

                std::fill(assigned.begin(), assigned.begin() + det_size, false);
                for (int i = 0; i < det_size; i++) {
                    ignored_threshold[i] = (img_dt[i * dt_cols + dt_cols - 1] < thresh);
                }
                delta_vec.clear();

                int tp = 0, fp = 0, fn = 0;
                double similarity = 0.0;

                constexpr int NO_DETECTION = -10000000;
                for (int i = 0; i < gt_size; i++) {
                    if (img_igt[i] == -1) continue;

                    int det_idx = -1;
                    float valid_detection = (float)NO_DETECTION;
                    float max_overlap = 0.0f;
                    bool assigned_ignored_det = false;

                    for (int j = 0; j < det_size; j++) {
                        if (img_idt[j] == -1 || assigned[j] || ignored_threshold[j])
                            continue;

                        float overlap_val = img_ov[j * gt_size + i];

                        if (overlap_val > (float)min_overlap
                            && (overlap_val > max_overlap || assigned_ignored_det)
                            && img_idt[j] == 0) {
                            max_overlap = overlap_val;
                            det_idx = j;
                            valid_detection = 1.0f;
                            assigned_ignored_det = false;
                        } else if (overlap_val > (float)min_overlap
                                   && valid_detection == (float)NO_DETECTION
                                   && img_idt[j] == 1) {
                            det_idx = j;
                            valid_detection = 1.0f;
                            assigned_ignored_det = true;
                        }
                    }

                    if (valid_detection == (float)NO_DETECTION && img_igt[i] == 0) {
                        fn++;
                    } else if (valid_detection != (float)NO_DETECTION
                               && (img_igt[i] == 1 || img_idt[det_idx] == 1)) {
                        assigned[det_idx] = true;
                    } else if (valid_detection != (float)NO_DETECTION) {
                        tp++;
                        if (compute_aos) {
                            float gt_alpha = img_gt[i * 5 + 4];
                            float dt_alpha = img_dt[det_idx * dt_cols + 4];
                            delta_vec.push_back(gt_alpha - dt_alpha);
                        }
                        assigned[det_idx] = true;
                    }
                }

                for (int i = 0; i < det_size; i++) {
                    if (!(assigned[i] || img_idt[i] == -1
                          || img_idt[i] == 1 || ignored_threshold[i]))
                        fp++;
                }

                int nstuff = 0;
                if (metric == 0 && dc_size > 0 && img_dc != nullptr) {
                    for (int dc_i = 0; dc_i < dc_size; dc_i++) {
                        for (int j = 0; j < det_size; j++) {
                            if (assigned[j] || img_idt[j] == -1
                                || img_idt[j] == 1 || ignored_threshold[j])
                                continue;

                            float dt_x1 = img_dt[j * dt_cols + 0];
                            float dt_y1 = img_dt[j * dt_cols + 1];
                            float dt_x2 = img_dt[j * dt_cols + 2];
                            float dt_y2 = img_dt[j * dt_cols + 3];
                            float dc_x1 = img_dc[dc_i * 4 + 0];
                            float dc_y1 = img_dc[dc_i * 4 + 1];
                            float dc_x2 = img_dc[dc_i * 4 + 2];
                            float dc_y2 = img_dc[dc_i * 4 + 3];

                            float iw = std::min(dt_x2, dc_x2) - std::max(dt_x1, dc_x1);
                            if (iw <= 0) continue;
                            float ih = std::min(dt_y2, dc_y2) - std::max(dt_y1, dc_y1);
                            if (ih <= 0) continue;

                            float dt_area = (dt_x2 - dt_x1) * (dt_y2 - dt_y1);
                            float ov_val = (dt_area > 0) ? (iw * ih) / dt_area : 0.0f;

                            if (ov_val > (float)min_overlap) {
                                assigned[j] = true;
                                nstuff++;
                            }
                        }
                    }
                }
                fp -= nstuff;

                if (compute_aos) {
                    if (tp > 0 || fp > 0) {
                        similarity = 0.0;
                        for (size_t k = 0; k < delta_vec.size(); k++)
                            similarity += (1.0 + cos((double)delta_vec[k])) / 2.0;
                    } else {
                        similarity = -1.0;
                    }
                }

                my_pr[t * 4 + 0] += tp;
                my_pr[t * 4 + 1] += fp;
                my_pr[t * 4 + 2] += fn;
                if (similarity != -1.0)
                    my_pr[t * 4 + 3] += similarity;
            } // end threshold loop
        } // end image loop
    } // end omp parallel

    // 合并所有线程的 PR
    for (int tid = 0; tid < num_threads; tid++) {
        double* src = tl_pr.data() + tid * num_thresholds * 4;
        for (int i = 0; i < num_thresholds * 4; i++) {
            pr_ptr[i] += src[i];
        }
    }

    return pr;
}

// ═══════════════════ Python 绑定 ═══════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "KITTI 评估 CUDA 加速扩展 (优化版)";

    // IoU
    m.def("iou2d_gpu", &iou2d_gpu, "2D Box IoU (CUDA)",
          py::arg("boxes"), py::arg("query_boxes"), py::arg("criterion") = -1);
    m.def("rotate_iou_gpu", &rotate_iou_gpu, "Rotated BEV Box IoU (CUDA)",
          py::arg("boxes"), py::arg("query_boxes"), py::arg("criterion") = -1);
    m.def("d3_box_overlap_gpu", &d3_box_overlap_gpu, "3D Box Overlap / IoU (CUDA)",
          py::arg("boxes"), py::arg("query_boxes"), py::arg("criterion") = -1);

    // 单图片 (兼容)
    m.def("compute_statistics_cpp", &compute_statistics_cpp,
          "Compute evaluation statistics (单图片)",
          py::arg("overlaps"), py::arg("gt_datas"), py::arg("dt_datas"),
          py::arg("ignored_gt"), py::arg("ignored_det"), py::arg("dc_bboxes"),
          py::arg("metric"), py::arg("min_overlap"), py::arg("thresh"),
          py::arg("compute_fp"), py::arg("compute_aos"));

    // 批量 API (高性能)
    m.def("batch_collect_thresholds", &batch_collect_thresholds,
          "Batch collect TP thresholds (first pass, all images at once)",
          py::arg("overlap"), py::arg("gt_datas"), py::arg("dt_datas"),
          py::arg("ignored_gt"), py::arg("ignored_det"), py::arg("dc_bboxes"),
          py::arg("gt_nums"), py::arg("dt_nums"), py::arg("dc_nums"),
          py::arg("metric"), py::arg("min_overlap"));
    m.def("batch_compute_pr", &batch_compute_pr,
          "Batch compute PR across all thresholds × images (second pass)",
          py::arg("overlap"), py::arg("gt_datas"), py::arg("dt_datas"),
          py::arg("ignored_gt"), py::arg("ignored_det"), py::arg("dc_bboxes"),
          py::arg("gt_nums"), py::arg("dt_nums"), py::arg("dc_nums"),
          py::arg("metric"), py::arg("min_overlap"),
          py::arg("thresholds"), py::arg("compute_aos"));
}
