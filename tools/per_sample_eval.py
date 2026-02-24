"""Per-sample evaluation for KITTI 3D object detection.

Computes per-sample detection quality scores using the existing KITTI eval code
(compute_statistics_jit with 3D IoU metric), then selects samples that show
the desired improvement pattern: baseline worst, DA3 medium, DA3+U best.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.datasets.kitti.kitti_eval_python import kitti_common as kitti
from lib.datasets.kitti.kitti_eval_python.eval import (
    bev_box_overlap,
    clean_data,
    compute_statistics_jit,
    d3_box_overlap,
    image_box_overlap,
)


def _compute_overlap_single(gt_anno: dict, dt_anno: dict, metric: int):
    """Compute IoU overlap matrix for a single sample (GT × DT).

    Much faster than calculate_iou_partly which batches cross-sample boxes.

    Args:
        gt_anno: Single sample GT annotation dict.
        dt_anno: Single sample DT annotation dict.
        metric: 0=bbox, 1=bev, 2=3d.

    Returns:
        Overlap matrix of shape (num_dt, num_gt).
    """
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    if num_gt == 0 or num_dt == 0:
        return np.zeros((num_dt, num_gt), dtype=np.float64)

    if metric == 0:
        overlap = image_box_overlap(
            gt_anno["bbox"].astype(np.float64),
            dt_anno["bbox"].astype(np.float64),
        )
        return overlap.T  # transpose to (num_dt, num_gt)
    elif metric == 1:
        gt_loc = gt_anno["location"][:, [0, 2]]
        gt_dims = gt_anno["dimensions"][:, [0, 2]]
        gt_rots = gt_anno["rotation_y"]
        gt_boxes = np.concatenate(
            [gt_loc, gt_dims, gt_rots[..., np.newaxis]], axis=1,
        )
        dt_loc = dt_anno["location"][:, [0, 2]]
        dt_dims = dt_anno["dimensions"][:, [0, 2]]
        dt_rots = dt_anno["rotation_y"]
        dt_boxes = np.concatenate(
            [dt_loc, dt_dims, dt_rots[..., np.newaxis]], axis=1,
        )
        return bev_box_overlap(dt_boxes, gt_boxes).astype(np.float64)
    elif metric == 2:
        gt_loc = gt_anno["location"]
        gt_dims = gt_anno["dimensions"]
        gt_rots = gt_anno["rotation_y"]
        gt_boxes = np.concatenate(
            [gt_loc, gt_dims, gt_rots[..., np.newaxis]], axis=1,
        )
        dt_loc = dt_anno["location"]
        dt_dims = dt_anno["dimensions"]
        dt_rots = dt_anno["rotation_y"]
        dt_boxes = np.concatenate(
            [dt_loc, dt_dims, dt_rots[..., np.newaxis]], axis=1,
        )
        return d3_box_overlap(dt_boxes, gt_boxes).astype(np.float64)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_per_sample_scores(
    gt_annos: list[dict],
    dt_annos: list[dict],
    current_class: int = 0,
    difficulty: int = 1,
    metric: int = 2,
    min_overlap: float = 0.7,
) -> list[float]:
    """Compute a detection quality F1 score for each sample.

    Uses the KITTI eval matching logic (greedy best-overlap) to compute
    TP, FP, FN per sample, then returns F1 = 2*TP / (2*TP + FP + FN).

    Args:
        gt_annos: List of ground-truth annotation dicts (one per sample).
        dt_annos: List of detection annotation dicts (one per sample).
        current_class: Class index (0=Car, 1=Pedestrian, 2=Cyclist).
        difficulty: KITTI difficulty (0=easy, 1=moderate, 2=hard).
        metric: Eval metric (0=bbox, 1=bev, 2=3d).
        min_overlap: Minimum IoU overlap threshold.

    Returns:
        List of per-sample scores in [0, 1]. Higher is better.
        Returns -1.0 for samples with no valid GT objects.
    """
    assert len(gt_annos) == len(dt_annos)

    scores: list[float] = []
    for i in range(len(gt_annos)):
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = clean_data(
            gt_annos[i], dt_annos[i], current_class, difficulty,
        )
        if num_valid_gt == 0:
            scores.append(-1.0)
            continue

        overlap = _compute_overlap_single(gt_annos[i], dt_annos[i], metric)

        dc_arr = (
            np.zeros((0, 4), dtype=np.float64)
            if not dc_bboxes
            else np.stack(dc_bboxes, 0).astype(np.float64)
        )
        gt_data = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1,
        )
        dt_data = np.concatenate(
            [
                dt_annos[i]["bbox"],
                dt_annos[i]["alpha"][..., np.newaxis],
                dt_annos[i]["score"][..., np.newaxis],
            ],
            1,
        )

        tp, fp, fn, _, _ = compute_statistics_jit(
            overlap,
            gt_data,
            dt_data,
            np.array(ignored_gt, dtype=np.int64),
            np.array(ignored_det, dtype=np.int64),
            dc_arr,
            metric,
            min_overlap=min_overlap,
            thresh=0.0,
            compute_fp=True,
            compute_aos=False,
        )

        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom > 0 else 0.0)

    return scores


def select_improvement_samples(
    gt_label_dir: Path,
    val_split_file: Path,
    baseline_data_dir: Path,
    da3_data_dir: Path,
    da3u_data_dir: Path,
    top_k: int,
    allowed_sample_ids: list[int] | None = None,
    current_class: int = 0,
    difficulty: int = 1,
    metric: int = 2,
) -> list[tuple[str, float, float, float]]:
    """Select samples where baseline is worst, DA3 middle, DA3+U best.

    Args:
        gt_label_dir: Path to KITTI GT label directory (label_2/).
        val_split_file: Path to val.txt with image IDs.
        baseline_data_dir: Detection outputs/data dir for baseline (noDA3).
        da3_data_dir: Detection outputs/data dir for DA3.
        da3u_data_dir: Detection outputs/data dir for DA3+Uncertainty.
        top_k: Maximum number of samples to return.
        allowed_sample_ids: Optional filter of allowed KITTI sample IDs.
        current_class: Class index for evaluation (default: 0=Car).
        difficulty: KITTI difficulty level (default: 1=moderate).
        metric: Eval metric (0=bbox, 1=bev, 2=3d). Use 1 for BEV to
            highlight depth estimation improvements from DA3.

    Returns:
        List of (sample_name, baseline_f1, da3_f1, da3u_f1) tuples,
        sorted by improvement magnitude (da3u - baseline), descending.
    """
    metric_names = {0: "bbox", 1: "BEV", 2: "3D"}
    with open(val_split_file, "r") as f:
        val_ids = [int(line.strip()) for line in f if line.strip()]

    if allowed_sample_ids is not None:
        allowed_set = set(allowed_sample_ids)
        val_ids = [idx for idx in val_ids if idx in allowed_set]

    gt_annos = kitti.get_label_annos(str(gt_label_dir), val_ids)
    baseline_annos = kitti.get_label_annos(str(baseline_data_dir), val_ids)
    da3_annos = kitti.get_label_annos(str(da3_data_dir), val_ids)
    da3u_annos = kitti.get_label_annos(str(da3u_data_dir), val_ids)

    scores_b = compute_per_sample_scores(
        gt_annos, baseline_annos, current_class, difficulty, metric=metric,
    )
    scores_d = compute_per_sample_scores(
        gt_annos, da3_annos, current_class, difficulty, metric=metric,
    )
    scores_u = compute_per_sample_scores(
        gt_annos, da3u_annos, current_class, difficulty, metric=metric,
    )

    candidates: list[tuple[str, float, float, float, float]] = []
    for i, val_id in enumerate(val_ids):
        sb, sd, su = scores_b[i], scores_d[i], scores_u[i]
        if sb < 0 or sd < 0 or su < 0:
            continue
        if sb < sd < su:
            improvement = su - sb
            candidates.append((f"{val_id:06d}", improvement, sb, sd, su))

    candidates.sort(key=lambda x: x[1], reverse=True)

    results = [(name, sb, sd, su) for name, _, sb, sd, su in candidates[:top_k]]

    print(f"Per-sample eval ({metric_names.get(metric, '?')}): "
          f"{len(candidates)} samples match "
          f"baseline < DA3 < DA3+U pattern (top {top_k} selected).")
    for rank, (name, sb, sd, su) in enumerate(results, start=1):
        print(f"  {rank:>3}. {name}  baseline={sb:.3f}  "
              f"DA3={sd:.3f}  DA3+U={su:.3f}  gap={su - sb:.3f}")

    return results
