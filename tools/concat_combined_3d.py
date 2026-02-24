"""Horizontally concatenate combined_3d images from three visualization folders.

This script reads same-named images from:
- vis_output_DA3/combined_3d
- vis_output_DA3_U/combined_3d
- vis_output_noDA3/combined_3d

Then it concatenates them left-to-right and saves results into a separate folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from per_sample_eval import select_improvement_samples


def _resize_to_height(image, target_height: int):
    """Resize an image to a target height while keeping aspect ratio."""
    current_height, current_width = image.shape[:2]
    if current_height == target_height:
        return image
    scaled_width = int(current_width * (target_height / current_height))
    return cv2.resize(image, (scaled_width, target_height), interpolation=cv2.INTER_LINEAR)


def _find_bev_split(image) -> int:
    """Find the row index where the BEV (bird's-eye-view) portion starts.

    The combined_3d image has a camera view on top and a BEV view at the bottom,
    separated by a dark band.  The BEV background is light gray (mean > 180).

    Returns:
        Row index of the first BEV row, or ``image.shape[0] // 2`` as fallback.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row_means = gray.mean(axis=1)
    img_h = image.shape[0]
    for y in range(img_h // 3, img_h):
        if row_means[y] > 180:
            return y
    return img_h // 2


def _find_green_bbox_in_bev(bev_image, legend_margin=40):
    """Find the bounding box of green-colored pixels inside a BEV sub-image.

    The top-left corner (legend area) is excluded to avoid the green legend
    square being counted as part of the detection boxes.

    Returns:
        Tuple (x1, y1, x2, y2) in BEV-local coordinates, or None.
    """
    hsv = cv2.cvtColor(bev_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Zero out the legend area in the top-left corner
    mask[:legend_margin, :legend_margin] = 0

    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    return (x, y, x + w, y + h)


def _get_bev_green_bbox(image):
    """Return the green bbox in BEV-local coordinates and the BEV split row.

    Returns:
        Tuple (bev_y0, (x1, y1, x2, y2)) or (bev_y0, None) if no green found.
    """
    bev_y0 = _find_bev_split(image)
    bev = image[bev_y0:]
    bbox = _find_green_bbox_in_bev(bev)
    return bev_y0, bbox


def _union_bboxes(bboxes):
    """Compute the union (smallest enclosing rectangle) of multiple bboxes.

    Args:
        bboxes: Iterable of (x1, y1, x2, y2) tuples. None entries are skipped.

    Returns:
        (x1, y1, x2, y2) or None if all inputs are None.
    """
    x1 = y1 = float("inf")
    x2 = y2 = float("-inf")
    found = False
    for bbox in bboxes:
        if bbox is None:
            continue
        found = True
        x1 = min(x1, bbox[0])
        y1 = min(y1, bbox[1])
        x2 = max(x2, bbox[2])
        y2 = max(y2, bbox[3])
    return (int(x1), int(y1), int(x2), int(y2)) if found else None


def _add_label_header(image, text: str, header_height: int = 56):
    """Add a black header strip with centered white text above the image."""
    img_h, img_w = image.shape[:2]
    header = np.zeros((header_height, img_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (img_w - tw) // 2
    ty = (header_height + th) // 2
    cv2.putText(header, text, (tx, ty), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return cv2.vconcat([header, image])


def _add_zoom_bubble(image, unified_bbox=None, padding=30, max_ratio=0.75):
    """Crop the green-box region from the BEV portion, zoom it, and overlay in the BEV bottom-left.

    Only the lower BEV part of the combined_3d image is used for bubble placement.
    The upper camera view is left untouched.

    Args:
        image: BGR combined_3d image (camera on top, BEV on bottom).
        unified_bbox: Pre-computed BEV-local bbox (x1, y1, x2, y2) shared across
            all variants.  If None, auto-detects from this image alone.
        padding: Pixels to expand around the bbox.
        max_ratio: Maximum ratio of the bubble size relative to BEV dimensions.

    Returns:
        Image with the zoom bubble overlaid on the BEV, or the original unchanged.
    """
    img_h, img_w = image.shape[:2]
    bev_y0 = _find_bev_split(image)
    bev = image[bev_y0:]
    bev_h, bev_w = bev.shape[:2]

    bbox = unified_bbox if unified_bbox is not None else _find_green_bbox_in_bev(bev)
    if bbox is None:
        return image

    gx1, gy1, gx2, gy2 = bbox

    # Expand with padding, clamped to BEV bounds
    cx1 = max(0, gx1 - padding)
    cy1 = max(0, gy1 - padding)
    cx2 = min(bev_w, gx2 + padding)
    cy2 = min(bev_h, gy2 + padding)

    crop = bev[cy1:cy2, cx1:cx2].copy()
    crop_h, crop_w = crop.shape[:2]
    if crop_h == 0 or crop_w == 0:
        return image

    # Scale factor: zoom to fill up to max_ratio of BEV, keeping aspect ratio
    max_bw = int(bev_w * max_ratio)
    max_bh = int(bev_h * max_ratio)
    scale = min(max_bw / crop_w, max_bh / crop_h)
    scale = max(scale, 1.0)  # at least 1x
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    # Bubble position: bottom-left of the BEV, in global image coordinates
    margin = 10
    bx1 = margin
    by2 = img_h - margin
    by1 = by2 - new_h
    bx2 = bx1 + new_w

    # Clamp to image bounds
    if bx2 > img_w:
        bx2 = img_w
    if by1 < bev_y0:
        by1 = bev_y0
    actual_w = bx2 - bx1
    actual_h = by2 - by1
    if actual_w <= 0 or actual_h <= 0:
        return image

    zoomed = cv2.resize(crop, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)

    result = image.copy()
    border_thickness = 2

    # Source region rectangle in global coordinates
    src_x1 = cx1
    src_y1 = bev_y0 + cy1
    src_x2 = cx2
    src_y2 = bev_y0 + cy2

    # Draw semi-transparent magnification overlay (connecting lines + source rect)
    overlay = result.copy()
    highlight_color = (0, 0, 255)  # red in BGR

    # Dark background behind bubble
    cv2.rectangle(overlay, (bx1 - 4, by1 - 4), (bx2 + 4, by2 + 4), (0, 0, 0), -1)

    # Source region rectangle on the BEV
    cv2.rectangle(overlay, (src_x1, src_y1), (src_x2, src_y2),
                  highlight_color, border_thickness)

    # Connecting lines: left-top of source -> left-top of bubble
    cv2.line(overlay, (src_x1, src_y1), (bx1, by1),
             highlight_color, border_thickness, cv2.LINE_AA)
    # Right-bottom of source -> right-bottom of bubble
    cv2.line(overlay, (src_x2, src_y2), (bx2, by2),
             highlight_color, border_thickness, cv2.LINE_AA)

    # Blend overlay (semi-transparent)
    cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)

    # Place zoomed crop (fully opaque on top)
    result[by1:by2, bx1:bx2] = zoomed

    # Border around bubble (fully opaque)
    cv2.rectangle(result, (bx1 - border_thickness, by1 - border_thickness),
                  (bx2 + border_thickness, by2 + border_thickness),
                  (0, 0, 255), border_thickness)

    # Small "ZOOM" label inside bubble top-left
    cv2.putText(result, "ZOOM", (bx1 + 4, by1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return result


def resolve_data_dir(config_path: Path, configs_root: Path, results_root: Path) -> Path:
    """Resolve one config YAML path to its latest ``outputs/data`` directory.

    Mapping rule:
    ``experiments/configs/<subdirs>/<name>.yaml`` ->
    ``experiments/results/<subdirs>/<name>/<latest_timestamp>/outputs/data``

    Raises:
        FileNotFoundError: If mapped paths or ``outputs/data`` cannot be found.
    """
    config_path = config_path.resolve()
    configs_root = configs_root.resolve()
    results_root = results_root.resolve()

    try:
        rel = config_path.relative_to(configs_root)
    except ValueError as error:
        raise FileNotFoundError(
            f"Config {config_path} is not under configs_root {configs_root}."
        ) from error

    run_root = results_root / rel.parent / rel.stem
    if not run_root.exists():
        raise FileNotFoundError(f"Mapped result directory does not exist: {run_root}")

    timestamp_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamp run dirs under: {run_root}")

    latest_data_dir = timestamp_dirs[-1] / "outputs" / "data"
    if not latest_data_dir.exists():
        raise FileNotFoundError(f"outputs/data not found: {latest_data_dir}")

    return latest_data_dir





def concat_combined_3d(
    da3_dir: Path,
    da3_u_dir: Path,
    no_da3_dir: Path,
    output_dir: Path,
    target_png_names: list[str] | None = None,
) -> None:
    """Concatenate same-named images from three directories into one output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    da3_files = sorted(path.name for path in da3_dir.glob("*.png"))
    da3_u_files = {path.name for path in da3_u_dir.glob("*.png")}
    no_da3_files = {path.name for path in no_da3_dir.glob("*.png")}

    common_files = [name for name in da3_files if name in da3_u_files and name in no_da3_files]
    if not common_files:
        raise RuntimeError("No common PNG files found across the three combined_3d folders.")

    if target_png_names is not None:
        common_file_set = set(common_files)
        common_files = [name for name in target_png_names if name in common_file_set]
        if not common_files:
            raise RuntimeError("No target top-K PNG files exist in all three combined_3d folders.")

    missing_in_da3_u = [name for name in da3_files if name not in da3_u_files]
    missing_in_no_da3 = [name for name in da3_files if name not in no_da3_files]
    if missing_in_da3_u:
        print(f"Warning: {len(missing_in_da3_u)} files missing in DA3_U folder.")
    if missing_in_no_da3:
        print(f"Warning: {len(missing_in_no_da3)} files missing in noDA3 folder.")

    saved_count = 0
    rank_width = max(2, len(str(len(common_files))))
    for file_name in common_files:
        da3_img = cv2.imread(str(da3_dir / file_name), cv2.IMREAD_COLOR)
        da3_u_img = cv2.imread(str(da3_u_dir / file_name), cv2.IMREAD_COLOR)
        no_da3_img = cv2.imread(str(no_da3_dir / file_name), cv2.IMREAD_COLOR)

        if da3_img is None or da3_u_img is None or no_da3_img is None:
            print(f"Warning: failed to read {file_name}, skipped.")
            continue

        target_height = min(da3_img.shape[0], da3_u_img.shape[0], no_da3_img.shape[0])
        da3_img = _resize_to_height(da3_img, target_height)
        da3_u_img = _resize_to_height(da3_u_img, target_height)
        no_da3_img = _resize_to_height(no_da3_img, target_height)

        # Compute green bbox in BEV for each image, then take the union
        # so all three zoom bubbles show the exact same region.
        bboxes = [
            _get_bev_green_bbox(no_da3_img)[1],
            _get_bev_green_bbox(da3_img)[1],
            _get_bev_green_bbox(da3_u_img)[1],
        ]
        unified_bbox = _union_bboxes(bboxes)

        # Apply zoom bubble to each image with the unified bbox
        no_da3_img = _add_zoom_bubble(no_da3_img, unified_bbox=unified_bbox)
        da3_img = _add_zoom_bubble(da3_img, unified_bbox=unified_bbox)
        da3_u_img = _add_zoom_bubble(da3_u_img, unified_bbox=unified_bbox)

        # Add label headers
        sample_id = Path(file_name).stem.replace("_combined", "")
        no_da3_img = _add_label_header(no_da3_img, f"Baseline  ({sample_id})")
        da3_img = _add_label_header(da3_img, f"DA3  ({sample_id})")
        da3_u_img = _add_label_header(da3_u_img, f"DA3 + Uncertainty  ({sample_id})")

        # Order: left=baseline(noDA3), middle=DA3, right=DA3+U
        merged = cv2.hconcat([no_da3_img, da3_img, da3_u_img])
        rank = saved_count + 1
        output_name = f"{rank:0{rank_width}d}_{file_name}"
        cv2.imwrite(str(output_dir / output_name), merged)
        saved_count += 1

    print(f"Done. Saved {saved_count} concatenated images to: {output_dir}")


def get_common_png_names(da3_dir: Path, da3_u_dir: Path, no_da3_dir: Path) -> list[str]:
    """Get PNG names existing in all three combined_3d folders."""
    da3_files = sorted(path.name for path in da3_dir.glob("*.png"))
    da3_u_files = {path.name for path in da3_u_dir.glob("*.png")}
    no_da3_files = {path.name for path in no_da3_dir.glob("*.png")}
    return [name for name in da3_files if name in da3_u_files and name in no_da3_files]


def select_topk_png_names(
    config_baseline: Path,
    config_da3: Path,
    config_da3u: Path,
    configs_root: Path,
    results_root: Path,
    gt_label_dir: Path,
    val_split_file: Path,
    top_k: int,
    allowed_png_names: list[str] | None = None,
    metric: int = 2,
) -> list[str]:
    """Select top-K samples where baseline is worst, DA3 middle, DA3+U best.

    Uses per-sample detection F1 scores from the KITTI eval code to
    find samples with a clear improvement ladder across the three configs.

    Args:
        metric: Eval metric (0=bbox, 1=bev, 2=3d). Use 1 (BEV) to
            focus on depth estimation improvements from DA3.
    """
    baseline_data = resolve_data_dir(config_baseline, configs_root, results_root)
    da3_data = resolve_data_dir(config_da3, configs_root, results_root)
    da3u_data = resolve_data_dir(config_da3u, configs_root, results_root)

    allowed_ids = None
    if allowed_png_names is not None:
        allowed_ids = [
            int(Path(name).stem.replace("_combined", ""))
            for name in allowed_png_names
        ]

    results = select_improvement_samples(
        gt_label_dir=gt_label_dir,
        val_split_file=val_split_file,
        baseline_data_dir=baseline_data,
        da3_data_dir=da3_data,
        da3u_data_dir=da3u_data,
        top_k=top_k,
        allowed_sample_ids=allowed_ids,
        metric=metric,
    )

    return [f"{name}_combined.png" for name, _, _, _ in results]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Concatenate combined_3d images from DA3/DA3_U/noDA3.")
    parser.add_argument(
        "--da3_dir",
        type=Path,
        default=Path("vis_output_DA3/combined_3d"),
        help="Directory containing DA3 combined_3d PNG images.",
    )
    parser.add_argument(
        "--da3_u_dir",
        type=Path,
        default=Path("vis_output_DA3_U/combined_3d"),
        help="Directory containing DA3_U combined_3d PNG images.",
    )
    parser.add_argument(
        "--no_da3_dir",
        type=Path,
        default=Path("vis_output/combined_3d"),
        help="Directory containing noDA3 combined_3d PNG images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("vis_output_compare/combined_3d"),
        help="Output directory to save concatenated PNG images.",
    )
    parser.add_argument(
        "--config_baseline",
        type=Path,
        default=Path("experiments/configs/monodle/kitti_no_da3.yaml"),
        help="Config YAML for the baseline (noDA3) experiment.",
    )
    parser.add_argument(
        "--config_da3",
        type=Path,
        default=Path("experiments/configs/monodle/kitti_da3.yaml"),
        help="Config YAML for the DA3 experiment.",
    )
    parser.add_argument(
        "--config_da3u",
        type=Path,
        default=Path("experiments/configs/monodle/kitti_da3_uncertainty.yaml"),
        help="Config YAML for the DA3 + Uncertainty experiment.",
    )
    parser.add_argument(
        "--configs_root",
        type=Path,
        default=Path("experiments/configs"),
        help="Configs root directory for mapping to experiments/results.",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("experiments/results"),
        help="Results root directory.",
    )
    parser.add_argument(
        "--gt_label_dir",
        type=Path,
        default=Path("data/KITTI/training/label_2"),
        help="Path to KITTI ground-truth label directory.",
    )
    parser.add_argument(
        "--val_split_file",
        type=Path,
        default=Path("data/KITTI/ImageSets/val.txt"),
        help="Path to val.txt with image IDs.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=150,
        help="Select top-K samples with best improvement pattern.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["3d", "bev", "bbox"],
        default="3d",
        help="Eval metric for sample selection. "
             "Use 'bev' to highlight DA3 depth estimation improvements.",
    )
    return parser.parse_args()


def main() -> None:
    """Run image concatenation entrypoint."""
    args = parse_args()
    common_png_names = get_common_png_names(
        da3_dir=args.da3_dir,
        da3_u_dir=args.da3_u_dir,
        no_da3_dir=args.no_da3_dir,
    )
    if not common_png_names:
        raise RuntimeError("No common PNG files found across the three combined_3d folders.")
    print(f"Found {len(common_png_names)} common PNG files across vis folders.")

    metric_map = {"bbox": 0, "bev": 1, "3d": 2}
    target_png_names = select_topk_png_names(
        config_baseline=args.config_baseline,
        config_da3=args.config_da3,
        config_da3u=args.config_da3u,
        configs_root=args.configs_root,
        results_root=args.results_root,
        gt_label_dir=args.gt_label_dir,
        val_split_file=args.val_split_file,
        top_k=args.top_k,
        allowed_png_names=common_png_names,
        metric=metric_map[args.metric],
    )
    concat_combined_3d(
        da3_dir=args.da3_dir,
        da3_u_dir=args.da3_u_dir,
        no_da3_dir=args.no_da3_dir,
        output_dir=args.output_dir,
        target_png_names=target_png_names,
    )


if __name__ == "__main__":
    main()