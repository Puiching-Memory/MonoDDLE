"""
KITTI 评估核心 (优化版)
======================

基于 C++/CUDA 加速的 KITTI 目标检测评估，主要优化:
- 批量 C++ 调用替代 Python per-image 循环
- 消除每次调用的 Tensor 创建/转换开销
- 保持与 kitti_eval_python 完全一致的计算结果

公共 API
--------
- :func:`get_official_eval_result`
- :func:`get_distance_eval_result`
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.cpp_extension import load

__all__ = [
    "get_official_eval_result",
    "get_distance_eval_result",
]

# ═══════════════════ JIT 编译 ═══════════════════

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
_SOURCES = [
    os.path.join(_SRC_DIR, "iou3d.cpp"),
    os.path.join(_SRC_DIR, "iou3d_kernel.cu"),
]

try:
    iou3d_cuda = load(
        name="iou3d_cuda",
        sources=_SOURCES,
        verbose=False,
        extra_cflags=["-O3", "-fopenmp"],
        extra_ldflags=["-lgomp"],
    )
except Exception as e:
    print(f"[kitti_eval] Error JIT compiling iou3d_cuda: {e}")
    raise

# ═══════════════════ 常量 ═══════════════════

CLASS_NAMES = ("car", "pedestrian", "cyclist", "van", "person_sitting", "truck")
MIN_HEIGHT = (40, 25, 25)
MAX_OCCLUSION = (0, 1, 2)
MAX_TRUNCATION = (0.15, 0.3, 0.5)
MAX_DISTANCE = (30, 50, 70)
N_SAMPLE_PTS = 41
DISTANCE_COVER = False

# ═══════════════════ IoU 计算 ═══════════════════


def image_box_overlap(boxes: np.ndarray, query_boxes: np.ndarray, criterion: int = -1) -> np.ndarray:
    """2D IoU (GPU)。"""
    N, K = boxes.shape[0], query_boxes.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=boxes.dtype)
    return iou3d_cuda.iou2d_gpu(
        torch.as_tensor(boxes, device="cuda", dtype=torch.float32).contiguous(),
        torch.as_tensor(query_boxes, device="cuda", dtype=torch.float32).contiguous(),
        criterion,
    ).cpu().numpy()


def bev_box_overlap(boxes: np.ndarray, qboxes: np.ndarray, criterion: int = -1) -> np.ndarray:
    """BEV rotated IoU (GPU)。"""
    N, K = boxes.shape[0], qboxes.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=boxes.dtype)
    return iou3d_cuda.rotate_iou_gpu(
        torch.as_tensor(boxes, device="cuda", dtype=torch.float32).contiguous(),
        torch.as_tensor(qboxes, device="cuda", dtype=torch.float32).contiguous(),
        criterion,
    ).cpu().numpy()


def d3_box_overlap(boxes: np.ndarray, qboxes: np.ndarray, criterion: int = -1) -> np.ndarray:
    """3D IoU (GPU)。"""
    N, K = boxes.shape[0], qboxes.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=boxes.dtype)
    return iou3d_cuda.d3_box_overlap_gpu(
        torch.as_tensor(boxes, device="cuda", dtype=torch.float32).contiguous(),
        torch.as_tensor(qboxes, device="cuda", dtype=torch.float32).contiguous(),
        criterion,
    ).cpu().numpy()


# ═══════════════════ 数据清洗 ═══════════════════


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class]
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0

    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]

        if gt_name == current_cls_name:
            valid_class = 1
        elif current_cls_name == "pedestrian" and gt_name == "person_sitting":
            valid_class = 0
        elif current_cls_name == "car" and gt_name == "van":
            valid_class = 0
        else:
            valid_class = -1

        ignore = (
            gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty]
            or gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty]
            or height <= MIN_HEIGHT[difficulty]
        )

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore and valid_class == 1):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])

    for i in range(num_dt):
        dt_name = dt_anno["name"][i].lower()
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if dt_name == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1

        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def clean_data_by_distance(gt_anno, dt_anno, current_class, difficulty):
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class]
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0

    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]

        if gt_name == current_cls_name:
            valid_class = 1
        elif current_cls_name == "pedestrian" and gt_name == "person_sitting":
            valid_class = 0
        elif current_cls_name == "car" and gt_name == "van":
            valid_class = 0
        else:
            valid_class = -1

        dis = float(np.linalg.norm(gt_anno["location"][i]))
        ignore = False

        if DISTANCE_COVER:
            if (gt_anno["occluded"][i] > MAX_OCCLUSION[2]
                    or gt_anno["truncated"][i] > MAX_TRUNCATION[2]
                    or height <= MIN_HEIGHT[2]
                    or dis > MAX_DISTANCE[difficulty]):
                ignore = True
        else:
            if difficulty == 0:
                if (gt_anno["occluded"][i] > MAX_OCCLUSION[2]
                        or gt_anno["truncated"][i] > MAX_TRUNCATION[2]
                        or height <= MIN_HEIGHT[2]
                        or dis > MAX_DISTANCE[difficulty]):
                    ignore = True
            else:
                if (gt_anno["occluded"][i] > MAX_OCCLUSION[2]
                        or gt_anno["truncated"][i] > MAX_TRUNCATION[2]
                        or height <= MIN_HEIGHT[2]
                        or dis > MAX_DISTANCE[difficulty]
                        or dis <= MAX_DISTANCE[difficulty - 1]):
                    ignore = True

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore and valid_class == 1):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])

    for i in range(num_dt):
        dt_name = dt_anno["name"][i].lower()
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if dt_name == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1

        if height < MIN_HEIGHT[2]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


# ═══════════════════ AP 辅助 ═══════════════════


def get_thresholds(scores: np.ndarray, num_gt: int, num_sample_pts: int = N_SAMPLE_PTS) -> List[float]:
    scores = np.sort(scores)[::-1].copy()
    current_recall = 0.0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        r_recall = (i + 2) / num_gt if i < len(scores) - 1 else l_recall
        if (r_recall - current_recall) < (current_recall - l_recall) and i < len(scores) - 1:
            continue
        thresholds.append(score)
        current_recall += 1.0 / (num_sample_pts - 1.0)
    return thresholds


def get_split_parts(num: int, num_part: int) -> List[int]:
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]
    parts = [same_part] * num_part
    if remain_num > 0:
        parts.append(remain_num)
    return parts


# ═══════════════════ IoU 分批计算 ═══════════════════


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.array([len(a["name"]) for a in dt_annos])
    total_gt_num = np.array([len(a["name"]) for a in gt_annos])
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_part = gt_annos[example_idx:example_idx + num_part]
        dt_part = dt_annos[example_idx:example_idx + num_part]

        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate([a["location"][:, [0, 2]] for a in gt_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in gt_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"][:, [0, 2]] for a in dt_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in dt_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError(f"unknown metric: {metric}")

        parted_overlaps.append(overlap_part)
        example_idx += num_part

    return parted_overlaps, split_parts, total_gt_num, total_dt_num


# ═══════════════════ 数据准备 ═══════════════════


def _prepare_data(gt_annos, dt_annos, current_class, difficulty, by_distance=False):
    gt_datas_list, dt_datas_list = [], []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares_ = [], [], []
    total_num_valid_gt = 0

    clean_fn = clean_data_by_distance if by_distance else clean_data

    for i in range(len(gt_annos)):
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = clean_fn(
            gt_annos[i], dt_annos[i], current_class, difficulty,
        )
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))

        dc_bboxes_np = (
            np.stack(dc_bboxes, 0).astype(np.float64) if dc_bboxes
            else np.zeros((0, 4), dtype=np.float64)
        )
        total_dc_num.append(dc_bboxes_np.shape[0])
        dontcares_.append(dc_bboxes_np)
        total_num_valid_gt += num_valid_gt

        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1,
        )
        dt_datas = np.concatenate(
            [dt_annos[i]["bbox"],
             dt_annos[i]["alpha"][..., np.newaxis],
             dt_annos[i]["score"][..., np.newaxis]], 1,
        )
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)

    return (
        gt_datas_list, dt_datas_list,
        ignored_gts, ignored_dets, dontcares_,
        np.array(total_dc_num, dtype=np.int64), total_num_valid_gt,
    )


# ═══════════════════ 主评估循环 (优化版) ═══════════════════


def eval_class(
    gt_annos, dt_annos, current_classes, difficultys, metric,
    min_overlaps, compute_aos=False, num_parts=50, by_distance=False,
    _prep_cache=None,
):
    """逐类别/难度/重叠阈值计算 precision / recall / AOS (批量 C++ 优化)。

    优化点:
      1. 预转换 overlap 为 Tensor (全局常量，避免 k-loop 内重复转换)
      2. 预拼接分区数据 + 预转 Tensor (每 (m,l) 仅一次，k-loop 复用)
      3. _prep_cache: 跨 metric 缓存 _prepare_data 结果 (由 do_eval 传入)
      4. tensor_cache: 跨 metric 缓存 precomputed_parts 张量 (避免重复转换)
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    parted_overlaps, _, total_dt_num, total_gt_num = rets

    # ── 优化①: 预转换 overlap → Tensor (整个函数只做一次) ──
    overlap_tensors = [
        torch.from_numpy(ov.astype(np.float32)).contiguous()
        for ov in parted_overlaps
    ]

    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)

    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            # ── 优化③: 跨 metric 缓存 _prepare_data ──
            _cache_key = (current_class, difficulty, by_distance)
            if _prep_cache is not None and _cache_key in _prep_cache:
                rets = _prep_cache[_cache_key]
            else:
                rets = _prepare_data(
                    gt_annos, dt_annos, current_class, difficulty,
                    by_distance=by_distance,
                )
                if _prep_cache is not None:
                    _prep_cache[_cache_key] = rets

            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets

            # ── 优化②+④: 预拼接分区 + 预转 Tensor, 跨 metric 缓存 ──
            _tensor_key = ("_tensors", current_class, difficulty, by_distance)
            if _prep_cache is not None and _tensor_key in _prep_cache:
                precomputed_parts = _prep_cache[_tensor_key]
            else:
                precomputed_parts = []
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_t = torch.from_numpy(
                        np.concatenate(gt_datas_list[idx:idx + num_part], 0).astype(np.float32)
                    ).contiguous()
                    dt_t = torch.from_numpy(
                        np.concatenate(dt_datas_list[idx:idx + num_part], 0).astype(np.float32)
                    ).contiguous()
                    dc_t = torch.from_numpy(
                        np.concatenate(dontcares[idx:idx + num_part], 0).astype(np.float32)
                    ).contiguous()
                    igt_t = torch.from_numpy(
                        np.concatenate(ignored_gts[idx:idx + num_part], 0)
                    ).contiguous()
                    idt_t = torch.from_numpy(
                        np.concatenate(ignored_dets[idx:idx + num_part], 0)
                    ).contiguous()
                    gn_t = torch.from_numpy(
                        total_gt_num[idx:idx + num_part].astype(np.int64)
                    ).contiguous()
                    dn_t = torch.from_numpy(
                        total_dt_num[idx:idx + num_part].astype(np.int64)
                    ).contiguous()
                    dcn_t = torch.from_numpy(
                        total_dc_num[idx:idx + num_part].astype(np.int64)
                    ).contiguous()
                    precomputed_parts.append((gt_t, dt_t, dc_t, igt_t, idt_t, gn_t, dn_t, dcn_t))
                    idx += num_part
                if _prep_cache is not None:
                    _prep_cache[_tensor_key] = precomputed_parts

            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                # ─── 第一阶段: 批量收集 TP 阈值 ───
                all_thresholds_list = []
                for j in range(len(split_parts)):
                    gt_t, dt_t, dc_t, igt_t, idt_t, gn_t, dn_t, dcn_t = precomputed_parts[j]

                    thresh_t = iou3d_cuda.batch_collect_thresholds(
                        overlap_tensors[j],
                        gt_t, dt_t, igt_t, idt_t, dc_t,
                        gn_t, dn_t, dcn_t,
                        metric, float(min_overlap),
                    )
                    if thresh_t.numel() > 0:
                        all_thresholds_list.append(thresh_t.numpy())

                if not all_thresholds_list:
                    continue
                all_thresholds_arr = np.concatenate(all_thresholds_list)
                thresholds_list = get_thresholds(all_thresholds_arr, total_num_valid_gt)
                thresholds_arr = np.array(thresholds_list, dtype=np.float32)

                if len(thresholds_arr) == 0:
                    continue

                # ─── 第二阶段: 批量计算 PR ───
                thresholds_t = torch.from_numpy(thresholds_arr).contiguous()
                pr_total = np.zeros([len(thresholds_arr), 4], dtype=np.float64)
                for j in range(len(split_parts)):
                    gt_t, dt_t, dc_t, igt_t, idt_t, gn_t, dn_t, dcn_t = precomputed_parts[j]

                    pr_part = iou3d_cuda.batch_compute_pr(
                        overlap_tensors[j],
                        gt_t, dt_t, igt_t, idt_t, dc_t,
                        gn_t, dn_t, dcn_t,
                        metric, float(min_overlap),
                        thresholds_t,
                        compute_aos,
                    )
                    pr_total += pr_part.numpy()

                # 计算 precision / recall
                for i in range(len(thresholds_arr)):
                    recall[m, l, k, i] = pr_total[i, 0] / (pr_total[i, 0] + pr_total[i, 2])
                    precision[m, l, k, i] = pr_total[i, 0] / (pr_total[i, 0] + pr_total[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr_total[i, 3] / (pr_total[i, 0] + pr_total[i, 1])

                # 单调化
                for i in range(len(thresholds_arr)):
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)

    return {"recall": recall, "precision": precision, "orientation": aos}


# ═══════════════════ mAP ═══════════════════


def get_mAP(prec):
    sums = sum(prec[..., i] for i in range(0, prec.shape[-1], 4))
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = sum(prec[..., i] for i in range(1, prec.shape[-1]))
    return sums / 40 * 100


# ═══════════════════ do_eval ═══════════════════


def do_eval(gt_annos, dt_annos, current_classes, min_overlaps,
            compute_aos=False, DIForDIS=True):
    difficultys = [0, 1, 2]
    by_distance = not DIForDIS

    # 优化: _prepare_data 不依赖 metric，共享缓存避免 3× 重复计算
    _prep_cache = {}

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
                     min_overlaps, compute_aos, by_distance=by_distance,
                     _prep_cache=_prep_cache)
    mAP_bbox = get_mAP(ret["precision"])
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                     min_overlaps, by_distance=by_distance,
                     _prep_cache=_prep_cache)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps, by_distance=by_distance,
                     _prep_cache=_prep_cache)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])

    return (mAP_bbox, mAP_bev, mAP_3d, mAP_aos,
            mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40)


# ═══════════════════ 公共 API ═══════════════════

_CLASS_TO_NAME = {0: "Car", 1: "Pedestrian", 2: "Cyclist",
                  3: "Van", 4: "Person_sitting", 5: "Truck"}
_NAME_TO_CLASS = {v: k for k, v in _CLASS_TO_NAME.items()}

_OVERLAP_07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                         [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                         [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
_OVERLAP_05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5],
                         [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                         [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
_MIN_OVERLAPS = np.stack([_OVERLAP_07, _OVERLAP_05], axis=0)


def _resolve_classes(current_classes):
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    result = []
    for cls in current_classes:
        if isinstance(cls, str):
            result.append(_NAME_TO_CLASS[cls])
        else:
            result.append(int(cls))
    return result


def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    current_classes_int = _resolve_classes(current_classes)
    min_overlaps = _MIN_OVERLAPS[:, :, current_classes_int]

    compute_aos = False
    for anno in dt_annos:
        if anno["alpha"].shape[0] != 0:
            if anno["alpha"][0] != -10:
                compute_aos = True
            break

    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes_int, min_overlaps, compute_aos, DIForDIS=True,
    )

    ret_dict = {}
    rich_data = []
    result = ""

    for j, curcls in enumerate(current_classes_int):
        cname = _CLASS_TO_NAME[curcls]
        for i in range(min_overlaps.shape[0]):
            overlap_str = "{:.2f}, {:.2f}, {:.2f}".format(*min_overlaps[i, :, j])
            block_data = {
                "class_name": cname,
                "overlap_str": overlap_str,
                "metrics": {"bbox": mAPbbox[j, :, i], "bev": mAPbev[j, :, i], "3d": mAP3d[j, :, i]},
                "metrics_R40": {"bbox": mAPbbox_R40[j, :, i], "bev": mAPbev_R40[j, :, i], "3d": mAP3d_R40[j, :, i]},
            }
            if compute_aos:
                block_data["metrics"]["aos"] = mAPaos[j, :, i]
                block_data["metrics_R40"]["aos"] = mAPaos_R40[j, :, i]
                if i == 0:
                    for d, dname in enumerate(("easy", "moderate", "hard")):
                        ret_dict[f"{cname}_aos_{dname}"] = mAPaos[j, d, 0]
                        ret_dict[f"{cname}_aos_{dname}_R40"] = mAPaos_R40[j, d, 0]
            rich_data.append(block_data)
            if i == 0:
                for suffix, arr, arr_r40 in [
                    ("3d", mAP3d, mAP3d_R40),
                    ("bev", mAPbev, mAPbev_R40),
                    ("image", mAPbbox, mAPbbox_R40),
                ]:
                    for d, dname in enumerate(("easy", "moderate", "hard")):
                        ret_dict[f"{cname}_{suffix}_{dname}"] = arr[j, d, 0]
                        ret_dict[f"{cname}_{suffix}_{dname}_R40"] = arr_r40[j, d, 0]

    return result, ret_dict, rich_data


def get_distance_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    current_classes_int = _resolve_classes(current_classes)
    min_overlaps = _MIN_OVERLAPS[:, :, current_classes_int]

    compute_aos = False
    for anno in dt_annos:
        if anno["alpha"].shape[0] != 0:
            if anno["alpha"][0] != -10:
                compute_aos = True
            break

    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes_int, min_overlaps, compute_aos, DIForDIS=False,
    )

    ret_dict = {}
    result = ""
    dist_names = ("30m", "50m", "70m")

    for j, curcls in enumerate(current_classes_int):
        cname = _CLASS_TO_NAME[curcls]
        for i in range(min_overlaps.shape[0]):
            if i == 0:
                for suffix, arr, arr_r40 in [
                    ("3d", mAP3d, mAP3d_R40),
                    ("bev", mAPbev, mAPbev_R40),
                    ("image", mAPbbox, mAPbbox_R40),
                ]:
                    for d, dn in enumerate(dist_names):
                        ret_dict[f"{cname}_{suffix}_{dn}"] = arr[j, d, 0]
                        ret_dict[f"{cname}_{suffix}_{dn}_R40"] = arr_r40[j, d, 0]

    return result, ret_dict
