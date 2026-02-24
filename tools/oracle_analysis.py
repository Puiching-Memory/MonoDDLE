"""Oracle analysis: replace decoder output heads with GT values one at a time.

For each oracle condition, runs model inference on the val set, replaces the
specified head outputs with ground truth at matched detection locations, then
evaluates 3D AP (R40) per depth bin and plots X=depth vs Y=AP_R40 curves.

Usage:
    python tools/oracle_analysis.py --config experiments/configs/monodle/kitti_da3.yaml

The script expects a trained checkpoint specified in the YAML tester section.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile

import numpy as np
import torch
import tqdm
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.utils import class2angle
from lib.helpers.decode_helper import (
    _nms,
    _topk,
    _transpose_and_gather_feat,
    decode_detections,
    extract_dets_from_outputs,
    get_heading_angle,
)
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.utils_helper import set_random_seed

import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti

# --- Use CUDA-accelerated eval (9.7x faster) ---
_cuda_ext_dir = os.path.join(
    ROOT_DIR, "lib", "datasets", "kitti", "kitti_eval_cuda")
if _cuda_ext_dir not in sys.path:
    sys.path.insert(0, _cuda_ext_dir)
from eval_cuda import get_official_eval_result


# ---------------------------------------------------------------------------
# Oracle modes: which heads to replace with GT
# ---------------------------------------------------------------------------
ORACLE_MODES = {
    "baseline": {
        "description": "No replacement (model predictions only)",
        "heads": [],
    },
    "w/ gt proj. center": {
        "description": "Replace 3D projected center offset with GT",
        "heads": ["offset_3d"],
    },
    "w/ gt depth": {
        "description": "Replace depth prediction with GT",
        "heads": ["depth"],
    },
    "w/ gt location": {
        "description": "Replace depth + 3D projected center (full 3D location) with GT",
        "heads": ["depth", "offset_3d"],
    },
    "w/ gt size_3d": {
        "description": "Replace 3D dimensions with GT",
        "heads": ["size_3d"],
    },
    "w/ gt heading": {
        "description": "Replace heading angle with GT",
        "heads": ["heading"],
    },
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Oracle analysis for MonoDDLE")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Oracle modes to run (default: all). Choose from: "
        + ", ".join(ORACLE_MODES.keys()),
    )
    parser.add_argument(
        "--depth_min", type=float, default=0.0, help="Minimum depth (m)"
    )
    parser.add_argument(
        "--depth_max", type=float, default=70.0, help="Maximum depth (m)"
    )
    parser.add_argument(
        "--depth_step", type=float, default=10.0, help="Depth bin width (m)"
    )
    parser.add_argument(
        "--category", default="Car", help="Object category for evaluation"
    )
    parser.add_argument(
        "--output", default=None, help="Output plot path (default: oracle_analysis.pdf)"
    )
    parser.add_argument(
        "--compare-configs", nargs="+", default=None,
        help="Compare oracle curves across multiple experiment configs",
    )
    parser.add_argument(
        "--compare-labels", nargs="+", default=None,
        help="Display labels for --compare-configs (default: config filename stems)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core: run inference and collect outputs + targets
# ---------------------------------------------------------------------------
def collect_inference_results(model, dataloader, device, max_objs=50):
    """Run inference and collect raw outputs and ground-truth targets.

    Returns:
        List of dicts, each containing 'outputs', 'targets', 'info' per batch.
    """
    model.eval()
    results = []
    with torch.no_grad():
        for inputs, targets, info in tqdm.tqdm(dataloader, desc="Inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Move outputs to CPU
            outputs_cpu = {k: v.detach().cpu() for k, v in outputs.items()}
            # targets are already numpy/tensor from dataloader
            results.append({
                "outputs": outputs_cpu,
                "targets": targets,
                "info": info,
            })
    return results


# ---------------------------------------------------------------------------
# Match detections to GT objects
# ---------------------------------------------------------------------------
def match_dets_to_gt(det_inds, gt_indices, gt_mask, feat_w):
    """Match each detected index to the nearest valid GT object.

    Args:
        det_inds: (K,) detected flattened indices in the feature map.
        gt_indices: (max_objs,) GT flattened indices.
        gt_mask: (max_objs,) boolean mask for valid GT objects.
        feat_w: width of the feature map.

    Returns:
        matched_gt: (K,) index into gt arrays for each detection, or -1.
    """
    # Convert flattened indices to (x, y)
    det_x = det_inds % feat_w
    det_y = det_inds // feat_w

    valid_idx = np.where(gt_mask)[0]
    if len(valid_idx) == 0:
        return np.full(len(det_inds), -1, dtype=np.int64)

    gt_x = gt_indices[valid_idx] % feat_w
    gt_y = gt_indices[valid_idx] // feat_w

    # Compute distance matrix: (K, num_valid_gt)
    dx = det_x[:, None].astype(float) - gt_x[None, :].astype(float)
    dy = det_y[:, None].astype(float) - gt_y[None, :].astype(float)
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # Match: for each detection, find the nearest GT
    nearest = np.argmin(dist, axis=1)
    nearest_dist = dist[np.arange(len(det_inds)), nearest]

    # Only accept matches within a reasonable distance (e.g., radius 5 pixels)
    max_match_dist = 5.0
    matched_gt = np.full(len(det_inds), -1, dtype=np.int64)
    close_enough = nearest_dist <= max_match_dist
    matched_gt[close_enough] = valid_idx[nearest[close_enough]]

    return matched_gt


# ---------------------------------------------------------------------------
# Replace heads in dets tensor with GT values
# ---------------------------------------------------------------------------
def replace_heads_with_gt(dets, inds, targets, heads_to_replace, batch_idx, feat_w):
    """Replace specified head columns in the dets tensor with GT values.

    Args:
        dets: (K, D) detection tensor for one image (numpy).
        inds: (K,) flattened feature map indices of detections.
        targets: target dict from dataloader.
        heads_to_replace: list of head names to replace.
        batch_idx: index in the batch.
        feat_w: feature map width.

    Returns:
        Modified dets array (copy).

    dets column layout (from extract_dets_from_outputs):
        0: cls_id, 1: score, 2: xs2d, 3: ys2d,
        4-5: size_2d, 6: depth, 7-30: heading(24),
        31-33: size_3d(3), 34: xs3d, 35: ys3d, 36: sigma
    """
    dets = dets.copy()
    K = dets.shape[0]

    # Get GT arrays for this image
    gt_indices = targets["indices"][batch_idx].numpy().astype(np.int64)
    gt_mask_2d = targets["mask_2d"][batch_idx].numpy().astype(bool)
    gt_mask_3d = targets["mask_3d"][batch_idx].numpy().astype(bool)

    # Use mask_2d for matching (it covers all valid objects)
    matched = match_dets_to_gt(inds, gt_indices, gt_mask_2d, feat_w)

    for head in heads_to_replace:
        if head == "depth":
            gt_depth = targets["depth"][batch_idx].numpy()  # (max_objs, 1)
            for k in range(K):
                gi = matched[k]
                if gi >= 0 and gt_mask_3d[gi]:
                    dets[k, 6] = gt_depth[gi, 0]

        elif head == "offset_3d":
            gt_offset_3d = targets["offset_3d"][batch_idx].numpy()  # (max_objs, 2)
            for k in range(K):
                gi = matched[k]
                if gi >= 0 and gt_mask_3d[gi]:
                    # The detection xs3d = xs + offset_3d_pred
                    # GT: offset_3d = center_3d - center_heatmap
                    # We need to set xs3d = center_heatmap_x + gt_offset_3d_x
                    center_hm_x = gt_indices[gi] % feat_w
                    center_hm_y = gt_indices[gi] // feat_w
                    dets[k, 34] = center_hm_x + gt_offset_3d[gi, 0]
                    dets[k, 35] = center_hm_y + gt_offset_3d[gi, 1]

        elif head == "size_3d":
            gt_size_3d = targets["size_3d"][batch_idx].numpy()  # (max_objs, 3)
            for k in range(K):
                gi = matched[k]
                if gi >= 0 and gt_mask_3d[gi]:
                    dets[k, 31:34] = gt_size_3d[gi]

        elif head == "heading":
            gt_heading_bin = targets["heading_bin"][batch_idx].numpy()  # (max_objs, 1)
            gt_heading_res = targets["heading_res"][batch_idx].numpy()  # (max_objs, 1)
            for k in range(K):
                gi = matched[k]
                if gi >= 0 and gt_mask_2d[gi]:
                    # Reconstruct the 24-dim heading vector
                    hbin = int(gt_heading_bin[gi, 0])
                    hres = gt_heading_res[gi, 0]
                    heading_vec = np.zeros(24, dtype=np.float32)
                    # Set the classification bin (one-hot like, high value)
                    heading_vec[hbin] = 10.0  # large logit so argmax picks it
                    # Set the residual at the correct bin
                    heading_vec[12 + hbin] = hres
                    dets[k, 7:31] = heading_vec

        elif head == "size_2d":
            gt_size_2d = targets["size_2d"][batch_idx].numpy()  # (max_objs, 2)
            for k in range(K):
                gi = matched[k]
                if gi >= 0 and gt_mask_2d[gi]:
                    dets[k, 4:6] = gt_size_2d[gi]

        elif head == "offset_2d":
            gt_offset_2d = targets["offset_2d"][batch_idx].numpy()  # (max_objs, 2)
            for k in range(K):
                gi = matched[k]
                if gi >= 0 and gt_mask_2d[gi]:
                    center_hm_x = gt_indices[gi] % feat_w
                    center_hm_y = gt_indices[gi] // feat_w
                    dets[k, 2] = center_hm_x + gt_offset_2d[gi, 0]
                    dets[k, 3] = center_hm_y + gt_offset_2d[gi, 1]

    return dets


# ---------------------------------------------------------------------------
# Run oracle for one mode
# ---------------------------------------------------------------------------
def run_oracle_inference(
    results_cache, heads_to_replace, dataloader, max_objs, threshold
):
    """Decode detections with specified GT head replacements.

    Returns:
        dict mapping img_id -> list of detection rows (KITTI format).
    """
    all_results = {}
    feat_w = dataloader.dataset.resolution[0] // dataloader.dataset.downsample

    for batch_data in results_cache:
        outputs = batch_data["outputs"]
        targets = batch_data["targets"]
        info = batch_data["info"]

        # Run standard extraction
        dets = extract_dets_from_outputs(outputs=outputs, K=max_objs)
        dets = dets.detach().cpu().numpy()

        # Get detected indices for matching
        heatmap = outputs["heatmap"]
        heatmap_proc = torch.clamp(heatmap.sigmoid(), min=1e-4, max=1 - 1e-4)
        heatmap_proc = _nms(heatmap_proc)
        _, inds, _, _, _ = _topk(heatmap_proc, K=max_objs)
        inds_np = inds.cpu().numpy()

        batch_size = dets.shape[0]
        for i in range(batch_size):
            if heads_to_replace:
                dets[i] = replace_heads_with_gt(
                    dets[i], inds_np[i], targets, heads_to_replace, i, feat_w
                )

        # Decode using standard pipeline
        calibs = [
            dataloader.dataset.get_calib(index) for index in info["img_id"]
        ]
        info_np = {key: val.detach().cpu().numpy() for key, val in info.items()}
        cls_mean_size = dataloader.dataset.cls_mean_size
        decoded = decode_detections(
            dets=dets,
            info=info_np,
            calibs=calibs,
            cls_mean_size=cls_mean_size,
            threshold=threshold,
        )
        all_results.update(decoded)

    return all_results


# ---------------------------------------------------------------------------
# Save results in KITTI format
# ---------------------------------------------------------------------------
def save_kitti_results(results, output_dir, class_name):
    """Write detection results in KITTI label format."""
    os.makedirs(output_dir, exist_ok=True)
    for img_id, preds in results.items():
        path = os.path.join(output_dir, "{:06d}.txt".format(img_id))
        with open(path, "w") as f:
            for det in preds:
                cls = class_name[int(det[0])]
                f.write("{} 0.0 0".format(cls))
                for j in range(1, len(det)):
                    f.write(" {:.2f}".format(det[j]))
                f.write("\n")


# ---------------------------------------------------------------------------
# Per-depth-bin evaluation
# ---------------------------------------------------------------------------
def filter_annos_by_depth(annos, depth_lo, depth_hi):
    """Create filtered copies of annotation dicts keeping only objects in [depth_lo, depth_hi).

    For GT: filter by location Z. For DT: filter by location Z.
    """
    filtered = []
    for anno in annos:
        if len(anno["name"]) == 0:
            filtered.append(copy.deepcopy(anno))
            continue

        depths = anno["location"][:, 2]  # Z coordinate = depth
        mask = (depths >= depth_lo) & (depths < depth_hi)

        new_anno = {}
        for key in anno:
            if isinstance(anno[key], np.ndarray):
                new_anno[key] = anno[key][mask]
            else:
                new_anno[key] = anno[key]
        filtered.append(new_anno)
    return filtered


def evaluate_per_depth_bin(
    gt_label_dir, dt_output_dir, val_ids, category, depth_bins
):
    """Evaluate 3D AP R40 per depth bin.

    Args:
        gt_label_dir: Path to GT label_2 directory.
        dt_output_dir: Path to detection output directory.
        val_ids: List of validation image IDs.
        category: Category name (e.g., 'Car').
        depth_bins: List of (depth_lo, depth_hi) tuples.

    Returns:
        List of AP_R40 values, one per depth bin.
    """
    gt_annos = kitti.get_label_annos(gt_label_dir, val_ids)
    dt_annos = kitti.get_label_annos(dt_output_dir, val_ids)

    test_id = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    cls_id = test_id[category]

    ap_per_bin = []
    for depth_lo, depth_hi in depth_bins:
        gt_filtered = filter_annos_by_depth(gt_annos, depth_lo, depth_hi)
        dt_filtered = filter_annos_by_depth(dt_annos, depth_lo, depth_hi)

        try:
            _, ret_dict = get_official_eval_result(
                gt_filtered, dt_filtered, cls_id
            )
            # Use moderate difficulty (index 1), 3D IoU @ 0.7
            # Use moderate difficulty 3D AP R40 key
            key = "{}_3d_moderate_R40".format(category)
            ap = ret_dict.get(key, None)
            if ap is None:
                # Fall back: compute directly
                ap = _extract_3d_ap_r40_moderate(gt_filtered, dt_filtered, cls_id)
        except Exception:
            ap = 0.0

        ap_per_bin.append(ap if ap is not None else 0.0)

    return ap_per_bin


def _extract_3d_ap_r40_moderate(gt_annos, dt_annos, cls_id):
    """Evaluate and extract the moderate 3D AP R40 from official eval."""
    from eval_cuda import (
        do_eval,
        get_mAP_R40,
    )

    overlap_0_7 = np.array(
        [
            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
        ]
    )
    overlap_0_5 = np.array(
        [
            [0.7, 0.5, 0.5, 0.7, 0.5, 0.5],
            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
        ]
    )
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)
    current_classes = [cls_id]
    min_overlaps = min_overlaps[:, :, current_classes]

    try:
        ret = do_eval(gt_annos, dt_annos, current_classes, min_overlaps, False)
        # ret = (mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40)
        mAP3d_R40 = ret[6]  # shape: [num_class, num_diff, num_minoverlap]
        # moderate = difficulty index 1, IoU@0.7 = minoverlap index 0
        return float(mAP3d_R40[0, 1, 0])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Compute overall AP_R40 (moderate)
# ---------------------------------------------------------------------------
def compute_overall_ap(gt_label_dir, dt_output_dir, val_ids, category):
    """Compute overall 3D AP R40 (moderate) for a category."""
    gt_annos = kitti.get_label_annos(gt_label_dir, val_ids)
    dt_annos = kitti.get_label_annos(dt_output_dir, val_ids)
    test_id = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    cls_id = test_id[category]
    return _extract_3d_ap_r40_moderate(gt_annos, dt_annos, cls_id)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_oracle_curves(
    depth_centers, ap_dict, overall_dict, output_path, category
):
    """Plot AP_R40 vs depth curves for all oracle modes.

    Args:
        depth_centers: array of depth bin center values.
        ap_dict: dict mapping mode_name -> list of AP values per bin.
        overall_dict: dict mapping mode_name -> overall AP value.
        output_path: path to save the plot.
        category: evaluated category name.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline": "#1f77b4",
        "w/ gt proj. center": "#ff7f0e",
        "w/ gt depth": "#2ca02c",
        "w/ gt location": "#d62728",
        "w/ gt size_3d": "#9467bd",
        "w/ gt heading": "#8c564b",
    }
    plt.figure(figsize=(8, 6))

    for mode_name, ap_values in ap_dict.items():
        overall = overall_dict.get(mode_name, 0.0)
        color = colors.get(mode_name, None)
        label = f"{mode_name} (overall AP$_{{40}}$: {overall:.2f})"
        plt.plot(depth_centers, ap_values, "-o", label=label, color=color,
                 linewidth=2, markersize=4)

    plt.xlabel("Depth", fontsize=14)
    plt.ylabel("AP$_{40}$", fontsize=14)
    plt.title(f"Oracle Analysis — {category} 3D AP R40 vs Depth", fontsize=14)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Run a single experiment config (reusable for comparison mode)
# ---------------------------------------------------------------------------
def run_single_config(config_path, modes, depth_bins, category, device, logger):
    """Run oracle analysis for one experiment config.

    Returns:
        dict: {mode_name: {"overall": float, "bin_aps": [float, ...]}}
    """
    import glob as _glob
    from torch.utils.data import DataLoader

    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    root_dir = cfg["dataset"].get("root_dir", "data/KITTI")
    if not os.path.isabs(root_dir):
        cfg["dataset"]["root_dir"] = os.path.normpath(
            os.path.join(ROOT_DIR, root_dir)
        )

    set_random_seed(cfg.get("random_seed", 444))

    # Build model
    model = build_model(cfg["model"])

    # Load checkpoint (with <timestamp> auto-resolution)
    ckpt_path = cfg["tester"]["checkpoint"]
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(ROOT_DIR, ckpt_path)
    if "<timestamp>" in ckpt_path:
        parent = ckpt_path.split("<timestamp>")[0]
        candidates = sorted(_glob.glob(os.path.join(parent, "*")))
        candidates = [c for c in candidates if os.path.isdir(c)]
        assert candidates, f"No experiment dirs in {parent}"
        ckpt_path = ckpt_path.replace(
            "<timestamp>", os.path.basename(candidates[-1])
        )
        logger.info(f"  Auto-resolved checkpoint: {ckpt_path}")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    load_checkpoint(
        model=model, optimizer=None, filename=ckpt_path,
        map_location=device, logger=logger,
    )
    model.to(device)
    model.eval()

    # Build val dataloader
    val_cfg = cfg["dataset"].copy()
    val_cfg["batch_size"] = cfg["dataset"].get("batch_size", 16)
    val_dataset = KITTI_Dataset(split="val", cfg=val_cfg)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_cfg["batch_size"],
        num_workers=val_cfg.get("num_workers", 4),
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    max_objs = val_dataset.max_objs
    threshold = cfg["tester"].get("threshold", 0.2)
    class_name = val_dataset.class_name
    gt_label_dir = val_dataset.label_dir
    val_ids = [int(idx) for idx in val_dataset.idx_list]

    # Inference (run model once)
    logger.info("  Running model inference ...")
    results_cache = collect_inference_results(model, val_loader, device, max_objs)

    # Free GPU memory for next config
    del model
    torch.cuda.empty_cache()

    # Evaluate each oracle mode
    results = {}
    for mode_name in modes:
        mode_cfg = ORACLE_MODES[mode_name]
        heads = mode_cfg["heads"]
        logger.info(f"  Evaluating: {mode_name}")

        decoded = run_oracle_inference(
            results_cache, heads, val_loader, max_objs, threshold
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "data")
            save_kitti_results(decoded, out_dir, class_name)
            for vid in val_ids:
                fp = os.path.join(out_dir, "{:06d}.txt".format(vid))
                if not os.path.exists(fp):
                    open(fp, "w").close()

            overall = compute_overall_ap(
                gt_label_dir, out_dir, val_ids, category
            )
            bin_aps = evaluate_per_depth_bin(
                gt_label_dir, out_dir, val_ids, category, depth_bins
            )

        results[mode_name] = {"overall": overall, "bin_aps": bin_aps}
        logger.info(f"    Overall AP_R40: {overall:.2f}")

    return results


# ---------------------------------------------------------------------------
# Comparison plotting with difference-area shading  (paper-quality, per-mode PNG)
# ---------------------------------------------------------------------------
def _setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "axes.grid": True,
    })
    return plt


def plot_comparison_curves(
    depth_centers, all_results, labels, output_dir, category, depth_bins
):
    """Plot per-mode comparison figures, each saved as an individual PNG.

    For each oracle mode, one figure shows curves from every experiment.
    The area between adjacent experiments is shaded green (gain) / red (loss).
    Delta annotations are placed *below* the plot in a dedicated text band
    so they never occlude data.

    Args:
        depth_centers: list of depth bin center values.
        all_results: dict  {label: {mode: {"overall", "bin_aps"}}}.
        labels: ordered experiment labels.
        output_dir: directory to write individual PNGs into.
        category: evaluated class name.
        depth_bins: list of (lo, hi) tuples.

    Returns:
        List of saved file paths.
    """
    plt = _setup_paper_style()
    os.makedirs(output_dir, exist_ok=True)

    modes = list(next(iter(all_results.values())).keys())

    # Palette
    EXP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    EXP_MARKERS = ["o", "s", "^", "D", "v"]
    GAIN_COLORS = ["#81C784", "#64B5F6", "#CE93D8"]
    LOSS_COLORS = ["#E57373", "#FFB74D", "#FFF176"]

    x = np.array(depth_centers)
    saved_paths = []

    for mode_name in modes:
        # Count delta lines for sizing the bottom text band
        n_deltas = len(labels) - 1
        bottom_margin = 0.06 + 0.04 * n_deltas  # fraction of fig height

        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        fig.subplots_adjust(bottom=bottom_margin + 0.12)

        prev_aps = None
        prev_label = None
        delta_texts = []

        for ci, label in enumerate(labels):
            aps = np.array(all_results[label][mode_name]["bin_aps"])
            overall = all_results[label][mode_name]["overall"]
            color = EXP_COLORS[ci % len(EXP_COLORS)]
            marker = EXP_MARKERS[ci % len(EXP_MARKERS)]
            ax.plot(
                x, aps, linestyle="-", marker=marker, color=color,
                zorder=3,
                label=f"{label}  (AP$_{{40}}$={overall:.1f})",
            )

            # Shade difference area between adjacent configs
            if prev_aps is not None:
                gc = GAIN_COLORS[(ci - 1) % len(GAIN_COLORS)]
                lc = LOSS_COLORS[(ci - 1) % len(LOSS_COLORS)]
                ax.fill_between(
                    x, prev_aps, aps,
                    where=(aps >= prev_aps),
                    color=gc, alpha=0.22, interpolate=True,
                )
                ax.fill_between(
                    x, prev_aps, aps,
                    where=(aps < prev_aps),
                    color=lc, alpha=0.22, interpolate=True,
                )

                # Compute signed area (trapezoidal)
                diff = aps - prev_aps
                gain = np.trapezoid(np.maximum(diff, 0), x)
                loss = np.trapezoid(np.maximum(-diff, 0), x)
                net = gain - loss
                sign_str = "+" if net >= 0 else "\u2212"
                clr = "#2E7D32" if net >= 0 else "#C62828"
                delta_texts.append((
                    f"\u0394({prev_label} \u2192 {label}):  "
                    f"\u2191 {gain:.0f}   \u2193 {loss:.0f}   "
                    f"net = {sign_str}{abs(net):.0f}",
                    clr,
                ))

            prev_aps = aps
            prev_label = label

        ax.set_title(f"{mode_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Depth (m)")
        ax.set_ylabel("AP$_{40}$")
        ax.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc")
        ax.set_xlim(x[0] - 2, x[-1] + 2)
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in")

        # --- Delta annotations BELOW the plot (in figure coordinates) ---
        for di, (txt, clr) in enumerate(delta_texts):
            y_pos = 0.02 + di * 0.04  # figure fraction, below axes
            fig.text(
                0.98, y_pos, txt,
                fontsize=7.5, fontfamily="monospace",
                ha="right", va="bottom", color=clr,
                transform=fig.transFigure,
            )

        # Save individual PNG
        safe_name = mode_name.replace("/", "_").replace(" ", "_")
        png_path = os.path.join(output_dir, f"{safe_name}.png")
        fig.savefig(png_path, bbox_inches="tight")
        saved_paths.append(png_path)
        plt.close(fig)
        print(f"  Saved: {png_path}")

    return saved_paths


# ---------------------------------------------------------------------------
# Write a LaTeX-style results table to a log file
# ---------------------------------------------------------------------------
def write_results_log(all_results, labels, modes, depth_bins, category,
                      log_path):
    """Write formatted results tables to a plain-text log file.

    Tables are aligned in a style suitable for direct inclusion in LaTeX
    documents via \\input or copy-paste into a \\begin{tabular} environment.
    """
    depth_headers = [f"{lo:.0f}--{hi:.0f}" for lo, hi in depth_bins]
    n_bins = len(depth_headers)

    lines = []
    lines.append(f"% Oracle Analysis Results — {category} 3D AP_{{R40}} (moderate)")
    lines.append(f"% Generated: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("")

    # ===== Table 1: Per-mode per-config (full breakdown) =====
    lines.append("%" + "=" * 78)
    lines.append("% Table: Per-depth-bin AP_R40")
    lines.append("%" + "=" * 78)
    col_spec = "ll" + "r" * (1 + n_bins)
    lines.append(f"% \\begin{{tabular}}{{{col_spec}}}")
    lines.append("% \\toprule")

    hdr = f"{'Method':<14s} & {'Oracle Mode':<22s} & {'Overall':>7s}"
    for dh in depth_headers:
        hdr += f" & {dh + 'm':>8s}"
    hdr += r" \\"
    lines.append(hdr)
    lines.append("% \\midrule")

    for li, label in enumerate(labels):
        for mi, mode_name in enumerate(modes):
            r = all_results[label][mode_name]
            method_col = label if mi == 0 else ""
            row = f"{method_col:<14s} & {mode_name:<22s} & {r['overall']:>7.2f}"
            for ap in r["bin_aps"]:
                row += f" & {ap:>8.2f}"
            row += r" \\"
            lines.append(row)
        if li < len(labels) - 1:
            lines.append("% \\midrule")

    lines.append("% \\bottomrule")
    lines.append("% \\end{tabular}")
    lines.append("")

    # ===== Table 2: Compact overall summary =====
    lines.append("%" + "=" * 78)
    lines.append("% Table: Overall AP_R40 Summary")
    lines.append("%" + "=" * 78)
    col_spec2 = "l" + "r" * len(modes)
    lines.append(f"% \\begin{{tabular}}{{{col_spec2}}}")
    lines.append("% \\toprule")

    hdr2 = f"{'Method':<14s}"
    for mode_name in modes:
        short = mode_name.replace("w/ gt ", "gt\\_")
        hdr2 += f" & {short:>14s}"
    hdr2 += r" \\"
    lines.append(hdr2)
    lines.append("% \\midrule")

    for label in labels:
        row2 = f"{label:<14s}"
        for mode_name in modes:
            row2 += f" & {all_results[label][mode_name]['overall']:>14.2f}"
        row2 += r" \\"
        lines.append(row2)

    lines.append("% \\bottomrule")
    lines.append("% \\end{tabular}")
    lines.append("")

    # ===== Table 3: Delta (improvement) between adjacent configs =====
    if len(labels) >= 2:
        lines.append("%" + "=" * 78)
        lines.append("% Table: Delta AP_R40 (config_i+1 - config_i)")
        lines.append("%" + "=" * 78)
        col_spec3 = "ll" + "r" * (1 + n_bins)
        lines.append(f"% \\begin{{tabular}}{{{col_spec3}}}")
        lines.append("% \\toprule")

        dhdr = f"{'Comparison':<28s} & {'Oracle Mode':<22s} & {'\\Delta Overall':>10s}"
        for dh in depth_headers:
            dhdr += f" & {'\\Delta ' + dh + 'm':>10s}"
        dhdr += r" \\"
        lines.append(dhdr)
        lines.append("% \\midrule")

        for li in range(len(labels) - 1):
            la, lb = labels[li], labels[li + 1]
            comp = f"{la} → {lb}"
            for mi, mode_name in enumerate(modes):
                ra = all_results[la][mode_name]
                rb = all_results[lb][mode_name]
                comp_col = comp if mi == 0 else ""
                d_overall = rb["overall"] - ra["overall"]
                sign_o = "+" if d_overall >= 0 else ""
                row3 = f"{comp_col:<28s} & {mode_name:<22s} & {sign_o}{d_overall:>9.2f}"
                for ai, bi in zip(ra["bin_aps"], rb["bin_aps"]):
                    d = bi - ai
                    sign_d = "+" if d >= 0 else ""
                    row3 += f" & {sign_d}{d:>9.2f}"
                row3 += r" \\"
                lines.append(row3)
            if li < len(labels) - 2:
                lines.append("% \\midrule")

        lines.append("% \\bottomrule")
        lines.append("% \\end{tabular}")
        lines.append("")

    content = "\n".join(lines) + "\n"
    with open(log_path, "w") as f:
        f.write(content)
    print(f"Results log saved to {log_path}")


# ---------------------------------------------------------------------------
# Multi-config comparison driver
# ---------------------------------------------------------------------------
def run_comparison(args):
    """Compare oracle analysis across multiple experiment configs."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)5s  %(message)s",
    )
    logger = logging.getLogger("oracle_comparison")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    configs = args.compare_configs
    labels = args.compare_labels
    if labels is None:
        labels = [os.path.splitext(os.path.basename(c))[0] for c in configs]
    assert len(labels) == len(configs), (
        f"Labels ({len(labels)}) must match configs ({len(configs)})"
    )

    modes = args.modes or list(ORACLE_MODES.keys())
    for m in modes:
        assert m in ORACLE_MODES, f"Unknown mode: {m}"

    bin_edges = np.arange(
        args.depth_min, args.depth_max + args.depth_step, args.depth_step
    )
    depth_bins = [
        (bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)
    ]
    depth_centers = [(lo + hi) / 2.0 for lo, hi in depth_bins]

    all_results = {}  # label -> {mode -> {overall, bin_aps}}
    for cfg_path, label in zip(configs, labels):
        logger.info("=" * 70)
        logger.info(f"Config: {label}  ({cfg_path})")
        logger.info("=" * 70)
        assert os.path.exists(cfg_path), f"Config not found: {cfg_path}"
        all_results[label] = run_single_config(
            cfg_path, modes, depth_bins, args.category, device, logger
        )

    # --- Output directory for individual PNGs ---
    output_dir = args.output or os.path.join(ROOT_DIR, "oracle_comparison")
    # If user gave a file path (e.g. foo.pdf), use its parent + stem as dir
    if os.path.splitext(output_dir)[1]:
        output_dir = os.path.splitext(output_dir)[0]

    # --- Plot individual comparison figures ---
    saved = plot_comparison_curves(
        depth_centers, all_results, labels, output_dir, args.category,
        depth_bins,
    )
    logger.info(f"Saved {len(saved)} figures to {output_dir}/")

    # --- Write results log file ---
    log_path = os.path.join(output_dir, "results.txt")
    write_results_log(
        all_results, labels, modes, depth_bins, args.category, log_path
    )

    # --- Print summary table to stdout ---
    _print_summary_table(all_results, labels, modes, depth_bins, args.category)


def _print_summary_table(all_results, labels, modes, depth_bins, category):
    """Print a concise comparison table to stdout."""
    depth_headers = [f"{lo:.0f}-{hi:.0f}m" for lo, hi in depth_bins]

    print("\n" + "=" * 100)
    print(f"Oracle Comparison Summary \u2014 {category}")
    print("=" * 100)
    header = f"{'Config':<14s} {'Mode':<22s} {'Overall':>8s}"
    for dh in depth_headers:
        header += f" {dh:>8s}"
    print(header)
    print("-" * len(header))
    for label in labels:
        for mode_name in modes:
            r = all_results[label][mode_name]
            row = f"{label:<14s} {mode_name:<22s} {r['overall']:>8.2f}"
            for ap in r["bin_aps"]:
                row += f" {ap:>8.2f}"
            print(row)
        print("-" * len(header))
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # --- Multi-config comparison mode ---
    if args.compare_configs:
        run_comparison(args)
        return

    # --- Single-config mode ---
    assert args.config is not None, (
        "Either --config or --compare-configs must be provided"
    )
    # Load config
    assert os.path.exists(args.config), f"Config not found: {args.config}"
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Resolve data root
    root_dir = cfg["dataset"].get("root_dir", "data/KITTI")
    if not os.path.isabs(root_dir):
        cfg["dataset"]["root_dir"] = os.path.normpath(
            os.path.join(ROOT_DIR, root_dir)
        )

    set_random_seed(cfg.get("random_seed", 444))

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)5s  %(message)s")
    logger = logging.getLogger("oracle_analysis")
    logger.setLevel(logging.INFO)

    # Build model
    model = build_model(cfg["model"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_path = cfg["tester"]["checkpoint"]
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(ROOT_DIR, ckpt_path)
    # Auto-resolve <timestamp> placeholder: find the latest experiment directory
    if "<timestamp>" in ckpt_path:
        import glob
        parent = ckpt_path.split("<timestamp>")[0]
        candidates = sorted(glob.glob(os.path.join(parent, "*")))
        candidates = [c for c in candidates if os.path.isdir(c)]
        assert candidates, f"No experiment directories found in {parent}"
        latest = candidates[-1]  # lexicographic sort → latest timestamp
        ts = os.path.basename(latest)
        ckpt_path = ckpt_path.replace("<timestamp>", ts)
        logger.info(f"Auto-resolved checkpoint: {ckpt_path}")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    load_checkpoint(
        model=model, optimizer=None, filename=ckpt_path,
        map_location=device, logger=logger,
    )
    model.to(device)
    model.eval()

    # Build val dataloader (no augmentation)
    val_cfg = cfg["dataset"].copy()
    val_cfg["batch_size"] = cfg["dataset"].get("batch_size", 16)
    val_dataset = KITTI_Dataset(split="val", cfg=val_cfg)
    from torch.utils.data import DataLoader

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_cfg["batch_size"],
        num_workers=val_cfg.get("num_workers", 4),
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    max_objs = val_dataset.max_objs
    threshold = cfg["tester"].get("threshold", 0.2)
    class_name = val_dataset.class_name
    gt_label_dir = val_dataset.label_dir
    val_ids = [int(idx) for idx in val_dataset.idx_list]

    # Step 1: Collect inference results (run model once)
    logger.info("Running model inference on validation set...")
    results_cache = collect_inference_results(model, val_loader, device, max_objs)

    # Step 2: Define depth bins
    bin_edges = np.arange(args.depth_min, args.depth_max + args.depth_step, args.depth_step)
    depth_bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    depth_centers = [(lo + hi) / 2.0 for lo, hi in depth_bins]

    # Step 3: Run each oracle mode
    modes = args.modes if args.modes else list(ORACLE_MODES.keys())
    ap_per_mode = {}
    overall_per_mode = {}

    for mode_name in modes:
        assert mode_name in ORACLE_MODES, f"Unknown mode: {mode_name}"
        mode_cfg = ORACLE_MODES[mode_name]
        heads = mode_cfg["heads"]

        logger.info(f"--- Oracle mode: {mode_name} ({mode_cfg['description']}) ---")

        # Run oracle inference
        decoded_results = run_oracle_inference(
            results_cache, heads, val_loader, max_objs, threshold
        )

        # Write to temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            output_data_dir = os.path.join(tmpdir, "data")
            save_kitti_results(decoded_results, output_data_dir, class_name)

            # Ensure all val images have a file (even empty)
            for vid in val_ids:
                fpath = os.path.join(output_data_dir, "{:06d}.txt".format(vid))
                if not os.path.exists(fpath):
                    open(fpath, "w").close()

            # Compute overall AP
            overall_ap = compute_overall_ap(
                gt_label_dir, output_data_dir, val_ids, args.category
            )
            overall_per_mode[mode_name] = overall_ap
            logger.info(f"  Overall 3D AP_R40 (moderate): {overall_ap:.2f}")

            # Compute per-depth-bin AP
            bin_aps = evaluate_per_depth_bin(
                gt_label_dir, output_data_dir, val_ids, args.category, depth_bins
            )
            ap_per_mode[mode_name] = bin_aps

            for (lo, hi), ap in zip(depth_bins, bin_aps):
                logger.info(f"  Depth [{lo:.0f}, {hi:.0f})m: AP_R40 = {ap:.2f}")

    # Step 4: Plot
    output_path = args.output or os.path.join(ROOT_DIR, "oracle_analysis.pdf")
    plot_oracle_curves(
        depth_centers, ap_per_mode, overall_per_mode, output_path, args.category
    )

    # Print summary table
    print("\n" + "=" * 80)
    print(f"Oracle Analysis Summary — {args.category}")
    print("=" * 80)
    header = f"{'Mode':<25s} {'Overall':>8s}"
    for lo, hi in depth_bins:
        header += f" {lo:.0f}-{hi:.0f}m".rjust(8)
    print(header)
    print("-" * len(header))
    for mode_name in modes:
        row = f"{mode_name:<25s} {overall_per_mode[mode_name]:>8.2f}"
        for ap in ap_per_mode[mode_name]:
            row += f" {ap:>8.2f}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()
