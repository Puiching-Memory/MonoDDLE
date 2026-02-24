"""
CUDA-accelerated KITTI 3D Object Detection Evaluation.

Faithfully reimplements the evaluation logic from kitti_eval_python/eval.py,
using C++/CUDA for IoU computation (the main bottleneck).

All numeric outputs match the original Python implementation exactly.
"""
import sys
import os
import numpy as np
import io as sysio

# Add the CUDA extension to path
_cuda_ext_dir = os.path.dirname(os.path.abspath(__file__))
if _cuda_ext_dir not in sys.path:
    sys.path.insert(0, _cuda_ext_dir)

import kitti_eval_cuda_ops as _C


def image_box_overlap(boxes, query_boxes, criterion=-1):
    """2D image box IoU. Delegates to C++. Matches original dtype."""
    import torch
    b = torch.from_numpy(np.ascontiguousarray(boxes, dtype=boxes.dtype))
    q = torch.from_numpy(np.ascontiguousarray(query_boxes, dtype=query_boxes.dtype))
    return _C.image_box_overlap(b, q, criterion).numpy()


def bev_box_overlap(boxes, qboxes, criterion=-1):
    """BEV rotated box IoU. Input cast to float32 (matching original), result cast to input dtype."""
    import torch
    # Original: boxes.astype(np.float32) inside rotate_iou_gpu_eval
    b = torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32))
    q = torch.from_numpy(np.ascontiguousarray(qboxes, dtype=np.float32))
    result = _C.bev_box_overlap(b, q, criterion).numpy()
    return result.astype(boxes.dtype)  # match original: .astype(np.float64)


def d3_box_overlap(boxes, qboxes, criterion=-1):
    """3D box overlap. BEV IoU (float32) + height overlap. Matches original dtype."""
    import torch
    b = torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32))
    q = torch.from_numpy(np.ascontiguousarray(qboxes, dtype=np.float32))
    result = _C.d3_box_overlap(b, q, criterion).numpy()
    return result.astype(boxes.dtype)


def get_thresholds(scores, num_gt, num_sample_pts=41):
    """Compute score thresholds for recall levels."""
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    """Filter GT and DT annotations by class and difficulty."""
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if gt_name == current_cls_name:
            valid_class = 1
        elif current_cls_name == "pedestrian" and "person_sitting" == gt_name:
            valid_class = 0
        elif current_cls_name == "car" and "van" == gt_name:
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if dt_anno["name"][i].lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)
    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def compute_statistics_jit(overlaps, gt_datas, dt_datas, ignored_gt, ignored_det,
                           dc_bboxes, metric, min_overlap, thresh=0,
                           compute_fp=False, compute_aos=False):
    """Compute TP/FP/FN statistics — delegates to C++ for speed."""
    import torch
    overlaps_t = torch.from_numpy(np.ascontiguousarray(overlaps, dtype=np.float64))
    gt_datas_t = torch.from_numpy(np.ascontiguousarray(gt_datas, dtype=np.float64))
    dt_datas_t = torch.from_numpy(np.ascontiguousarray(dt_datas, dtype=np.float64))
    ignored_gt_t = torch.from_numpy(np.ascontiguousarray(
        ignored_gt if isinstance(ignored_gt, np.ndarray) else np.array(ignored_gt, dtype=np.int64)))
    ignored_det_t = torch.from_numpy(np.ascontiguousarray(
        ignored_det if isinstance(ignored_det, np.ndarray) else np.array(ignored_det, dtype=np.int64)))
    dc_bboxes_t = torch.from_numpy(np.ascontiguousarray(dc_bboxes, dtype=np.float64))

    tp, fp, fn, similarity, thresholds_out = _C.compute_statistics_jit(
        overlaps_t, gt_datas_t, dt_datas_t,
        ignored_gt_t, ignored_det_t, dc_bboxes_t,
        int(metric), float(min_overlap), float(thresh),
        bool(compute_fp), bool(compute_aos))
    return tp, fp, fn, similarity, thresholds_out.numpy()


def get_split_parts(num, num_part):
    """Split num into num_part roughly equal parts."""
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


def fused_compute_statistics(overlaps, pr, gt_nums, dt_nums, dc_nums,
                             gt_datas, dt_datas, dontcares,
                             ignored_gts, ignored_dets,
                             metric, min_overlap, thresholds,
                             compute_aos=False):
    """Fused TP/FP/FN accumulation — delegates to C++ for speed."""
    import torch
    overlaps_t = torch.from_numpy(np.ascontiguousarray(overlaps, dtype=np.float64))
    # pr shares memory with numpy — C++ modifies it in-place
    pr_t = torch.from_numpy(np.ascontiguousarray(pr))
    gt_nums_t = torch.from_numpy(np.ascontiguousarray(gt_nums, dtype=np.int64))
    dt_nums_t = torch.from_numpy(np.ascontiguousarray(dt_nums, dtype=np.int64))
    dc_nums_t = torch.from_numpy(np.ascontiguousarray(dc_nums, dtype=np.int64))
    gt_datas_t = torch.from_numpy(np.ascontiguousarray(gt_datas, dtype=np.float64))
    dt_datas_t = torch.from_numpy(np.ascontiguousarray(dt_datas, dtype=np.float64))
    dontcares_t = torch.from_numpy(np.ascontiguousarray(dontcares, dtype=np.float64))
    ignored_gts_t = torch.from_numpy(np.ascontiguousarray(ignored_gts, dtype=np.int64))
    ignored_dets_t = torch.from_numpy(np.ascontiguousarray(ignored_dets, dtype=np.int64))
    thresholds_t = torch.from_numpy(np.ascontiguousarray(thresholds, dtype=np.float64))

    _C.fused_compute_statistics(
        overlaps_t, pr_t,
        gt_nums_t, dt_nums_t, dc_nums_t,
        gt_datas_t, dt_datas_t, dontcares_t,
        ignored_gts_t, ignored_dets_t,
        int(metric), float(min_overlap),
        thresholds_t, bool(compute_aos))
    # Copy result back to original numpy array (in case ascontiguousarray made a copy)
    np.copyto(pr, pr_t.numpy())


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """
    Fast IoU computation with CUDA acceleration.
    metric: 0=bbox, 1=bev, 2=3d
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part

    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    """Prepare data for evaluation of one class at one difficulty."""
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = clean_data(
            gt_annos[i], dt_annos[i], current_class, difficulty)
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos, dt_annos, current_classes, difficultys, metric,
               min_overlaps, compute_aos=False, num_parts=50,
               _prepare_cache=None):
    """KITTI eval. Support 2d/bev/3d/aos eval.
    _prepare_cache: optional dict keyed by (class, difficulty) to avoid redundant _prepare_data calls.
    """
    import torch
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            cache_key = (current_class, difficulty)
            if _prepare_cache is not None and cache_key in _prepare_cache:
                rets = _prepare_cache[cache_key]
            else:
                rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
                if _prepare_cache is not None:
                    _prepare_cache[cache_key] = rets
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets

            # Pre-convert all per-image data to torch tensors once
            overlaps_t = [torch.from_numpy(np.ascontiguousarray(o, dtype=np.float64)) for o in overlaps]
            gt_datas_t = [torch.from_numpy(np.ascontiguousarray(g, dtype=np.float64)) for g in gt_datas_list]
            dt_datas_t = [torch.from_numpy(np.ascontiguousarray(d, dtype=np.float64)) for d in dt_datas_list]
            ignored_gts_t = [torch.from_numpy(np.ascontiguousarray(
                ig if isinstance(ig, np.ndarray) else np.array(ig, dtype=np.int64))) for ig in ignored_gts]
            ignored_dets_t = [torch.from_numpy(np.ascontiguousarray(
                id_ if isinstance(id_, np.ndarray) else np.array(id_, dtype=np.int64))) for id_ in ignored_dets]
            dontcares_t = [torch.from_numpy(np.ascontiguousarray(dc, dtype=np.float64)) for dc in dontcares]

            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                # Batched threshold collection via C++ with OpenMP
                thresholdss_t = _C.collect_thresholds(
                    overlaps_t, gt_datas_t, dt_datas_t,
                    ignored_gts_t, ignored_dets_t, dontcares_t,
                    int(metric), float(min_overlap))
                thresholdss = thresholdss_t.numpy()
                thresholds_arr = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds_arr = np.array(thresholds_arr)
                pr = np.zeros([len(thresholds_arr), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j], pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part, dt_datas_part, dc_datas_part,
                        ignored_gts_part, ignored_dets_part,
                        metric, min_overlap=min_overlap,
                        thresholds=thresholds_arr,
                        compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds_arr)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds_arr)):
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    return {"recall": recall, "precision": precision, "orientation": aos}


def get_mAP(prec):
    """11-point mAP."""
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    """40-point mAP."""
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def print_str(value, *arg, sstream=None):
    """Print to string."""
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval(gt_annos, dt_annos, current_classes, min_overlaps,
            compute_aos=False, PR_detail_dict=None):
    """Run KITTI evaluation for bbox/bev/3d/aos."""
    difficultys = [0, 1, 2]
    # Shared cache for _prepare_data — keyed by (class, difficulty), independent of metric
    _prepare_cache = {}

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
                     min_overlaps, compute_aos, _prepare_cache=_prepare_cache)
    mAP_bbox = get_mAP(ret["precision"])
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bbox'] = ret['precision']

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])
        if PR_detail_dict is not None:
            PR_detail_dict['aos'] = ret['orientation']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                     min_overlaps, _prepare_cache=_prepare_cache)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['bev'] = ret['precision']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps, _prepare_cache=_prepare_cache)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']

    return (mAP_bbox, mAP_bev, mAP_3d, mAP_aos,
            mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40)


def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    """
    Official KITTI evaluation result.
    Format matches original eval.py exactly.
    """
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)
    class_to_name = {
        0: 'Car', 1: 'Pedestrian', 2: 'Cyclist',
        3: 'Van', 4: 'Person_sitting', 5: 'Truck'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]

    result = ''
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break

    (mAPbbox, mAPbev, mAP3d, mAPaos,
     mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40) = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos,
        PR_detail_dict=PR_detail_dict)

    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
                                 f"{mAPbbox[j, 1, i]:.4f}, "
                                 f"{mAPbbox[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
                                 f"{mAPbev[j, 1, i]:.4f}, "
                                 f"{mAPbev[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
                                 f"{mAP3d[j, 1, i]:.4f}, "
                                 f"{mAP3d[j, 2, i]:.4f}"))
            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
                                     f"{mAPaos[j, 1, i]:.2f}, "
                                     f"{mAPaos[j, 2, i]:.2f}"))
                if i == 0:
                    ret_dict['%s_aos_easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
                    ret_dict['%s_aos_moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
                    ret_dict['%s_aos_hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 1, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
                                 f"{mAPbev_R40[j, 1, i]:.4f}, "
                                 f"{mAPbev_R40[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
                                 f"{mAP3d_R40[j, 1, i]:.4f}, "
                                 f"{mAP3d_R40[j, 2, i]:.4f}"))
            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
                                     f"{mAPaos_R40[j, 1, i]:.2f}, "
                                     f"{mAPaos_R40[j, 2, i]:.2f}"))
                if i == 0:
                    ret_dict['%s_aos_easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
                    ret_dict['%s_aos_moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
                    ret_dict['%s_aos_hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]
            if i == 0:
                ret_dict['%s_3d_easy' % class_to_name[curcls]] = mAP3d[j, 0, 0]
                ret_dict['%s_3d_moderate' % class_to_name[curcls]] = mAP3d[j, 1, 0]
                ret_dict['%s_3d_hard' % class_to_name[curcls]] = mAP3d[j, 2, 0]
                ret_dict['%s_bev_easy' % class_to_name[curcls]] = mAPbev[j, 0, 0]
                ret_dict['%s_bev_moderate' % class_to_name[curcls]] = mAPbev[j, 1, 0]
                ret_dict['%s_bev_hard' % class_to_name[curcls]] = mAPbev[j, 2, 0]
                ret_dict['%s_image_easy' % class_to_name[curcls]] = mAPbbox[j, 0, 0]
                ret_dict['%s_image_moderate' % class_to_name[curcls]] = mAPbbox[j, 1, 0]
                ret_dict['%s_image_hard' % class_to_name[curcls]] = mAPbbox[j, 2, 0]
                ret_dict['%s_3d_easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
                ret_dict['%s_3d_moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
                ret_dict['%s_3d_hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
                ret_dict['%s_bev_easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
                ret_dict['%s_bev_moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
                ret_dict['%s_bev_hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
                ret_dict['%s_image_easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
                ret_dict['%s_image_moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
                ret_dict['%s_image_hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]

    return result, ret_dict
