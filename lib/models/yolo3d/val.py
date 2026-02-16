"""
Detection3DValidator: Extends DetectionValidator with KITTI 3D evaluation.

Supports standard NMS (YOLOv8/YOLO11) and end2end NMS-free (YOLO26) models.

After NMS/topk, extracts the 3D predictions from the 'extra' field, decodes them
to 3D bounding boxes (depth, dimensions, heading, location), and runs the
KITTI official evaluation for 2D + 3D metrics.

Integrates MonoDDLE-style evaluation:
  - Saves all predictions in KITTI format during validation
  - Runs KITTI official 3D evaluation (AP_3D, AP_BEV, etc.)
  - Generates 2D/3D bbox visualizations
"""

import os
import cv2
import numpy as np
import torch
from copy import copy
from pathlib import Path

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, RANK

from lib.datasets.utils import class2angle
from lib.datasets.kitti.kitti_utils import Calibration


class Detection3DValidator(DetectionValidator):
    """Validator for YOLO3D monocular 3D object detection on KITTI."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "detect3d"
        self.cls_mean_size = None
        self.num_heading_bin = 12
        # Storage for KITTI 3D evaluation
        self._kitti_results_dir = None
        self._kitti_3d_metrics = {}
        self._vis_batches = []  # store first few batches for visualization
        self._max_vis_batches = 2  # default, can be overridden by trainer
        self._current_epoch = None  # set by trainer via validate()

    def preprocess(self, batch):
        """Preprocess batch — images are already [0,1] from KITTI3DDataset,
        so we must NOT divide by 255 again (the parent DetectionValidator does)."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        return batch

    def init_metrics(self, model):
        """Initialize metrics. Load cls_mean_size from model if available."""
        super().init_metrics(model)
        if hasattr(model, "cls_mean_size"):
            self.cls_mean_size = model.cls_mean_size
        else:
            self.cls_mean_size = np.zeros((3, 3), dtype=np.float32)

        if hasattr(model, "model") and hasattr(model.model[-1], "num_heading_bin"):
            self.num_heading_bin = model.model[-1].num_heading_bin

        # Setup KITTI results directory
        self._kitti_results_dir = os.path.join(str(self.save_dir), 'kitti_eval', 'data')
        os.makedirs(self._kitti_results_dir, exist_ok=True)
        self._kitti_3d_metrics = {}
        self._vis_batches = []

    def postprocess(self, preds):
        """
        Apply NMS and extract 3D predictions from the 'extra' field.

        After standard NMS, the predictions dict has:
          - bboxes: [N, 4] (xyxy in image coords)
          - conf: [N]
          - cls: [N]
          - extra: [N, n3d] (raw 3D predictions passed through NMS)

        We reshape/decode the 3D predictions here.
        """
        preds = super().postprocess(preds)

        for pred in preds:
            extra = pred.get("extra", None)
            if extra is not None and extra.numel() > 0:
                # extra contains the decoded mono3d output:
                # [depth, sigma, x3d, y3d, size3d(3), heading(24)]
                pred["mono3d"] = extra
            else:
                pred["mono3d"] = torch.zeros(0, 31, device=pred["bboxes"].device)

        return preds

    def update_metrics(self, preds, batch):
        """
        Update metrics and save 3D predictions in KITTI format.
        
        Extends parent's update_metrics to also:
        1. Save each batch's 3D detections as KITTI-format .txt files
        2. Collect first few batches for visualization
        """
        # Standard 2D metrics update
        super().update_metrics(preds, batch)

        # Save 3D results in KITTI format for official evaluation
        if self._kitti_results_dir is not None and RANK in {-1, 0}:
            self.save_3d_results(preds, batch, self._kitti_results_dir)

        # Save first batch for visualization
        if self._max_vis_batches > 0 and len(self._vis_batches) < self._max_vis_batches and RANK in {-1, 0}:
            self._vis_batches.append((preds, batch))

    def _process_batch(self, preds, batch):
        """
        Process a batch for metrics. For now, we only evaluate 2D detection metrics
        through the parent class. Full 3D evaluation is done in finalize_metrics().
        """
        return super()._process_batch(preds, batch)

    def finalize_metrics(self, *args, **kwargs):
        """
        Finalize metrics: run KITTI official 3D evaluation and visualization.
        
        This is called after all validation batches have been processed.
        """
        super().finalize_metrics(*args, **kwargs)

        if RANK not in {-1, 0}:
            return

        # --- Run KITTI Official 3D Evaluation ---
        if self._kitti_results_dir is not None and self.dataloader is not None:
            dataset = self.dataloader.dataset
            try:
                LOGGER.info("Running KITTI official 3D evaluation...")
                self._kitti_3d_metrics = dataset.eval(
                    results_dir=self._kitti_results_dir,
                    logger=_UltralyticsLoggerAdapter(),
                    epoch=self._current_epoch,
                )
            except Exception as e:
                LOGGER.warning(f"KITTI 3D evaluation failed: {e}")
                import traceback
                traceback.print_exc()

        # --- Visualization ---
        if self._vis_batches and self.dataloader is not None:
            try:
                vis_dir = os.path.join(str(self.save_dir), 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                self._visualize_3d_results(vis_dir)
                LOGGER.info(f"3D visualizations saved to {vis_dir}")
            except Exception as e:
                LOGGER.warning(f"Visualization failed: {e}")
                import traceback
                traceback.print_exc()

    def _visualize_3d_results(self, output_dir):
        """
        Generate 2D/3D box visualizations for the stored validation batches.
        
        Draws:
          - Ground truth 3D boxes in blue
          - Predicted 3D boxes in green
          - 2D detection boxes overlay
        """
        dataset = self.dataloader.dataset
        class_names = dataset.class_name
        cls_mean_size = dataset.cls_mean_size
        threshold = 0.2

        for batch_idx, (preds_list, batch) in enumerate(self._vis_batches):
            img_ids = batch.get("img_id", [])
            images = batch.get("img", None)
            ratio_pads = batch.get("ratio_pad", None)

            for i, pred in enumerate(preds_list):
                if i >= len(img_ids):
                    break

                img_id = int(img_ids[i]) if not isinstance(img_ids[i], int) else img_ids[i]
                calib = dataset.get_calib(img_id)

                # Get scale ratio: network input space -> original image space
                if ratio_pads is not None and i < len(ratio_pads):
                    ratio_w, ratio_h = ratio_pads[i][0]
                else:
                    ratio_w, ratio_h = 1.0, 1.0

                # Load original image
                try:
                    from PIL import Image as PILImage
                    img_file = os.path.join(dataset.image_dir, '%06d.png' % img_id)
                    if os.path.exists(img_file):
                        img_pil = PILImage.open(img_file).convert('RGB')
                        img_base = np.array(img_pil)
                        img_base = cv2.cvtColor(img_base, cv2.COLOR_RGB2BGR)
                    else:
                        continue
                except Exception:
                    continue

                # --- 2D Box Visualization ---
                img_2d = img_base.copy()

                # Draw GT 2D boxes
                try:
                    gt_objects = dataset.get_label(img_id)
                    for obj in gt_objects:
                        if obj.cls_type == 'DontCare':
                            continue
                        if obj.cls_type not in class_names:
                            continue
                        x1, y1, x2, y2 = map(int, obj.box2d)
                        cv2.rectangle(img_2d, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for GT
                        cv2.putText(img_2d, obj.cls_type, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                except Exception:
                    gt_objects = []

                # Draw predicted 2D boxes
                bboxes = pred["bboxes"].cpu().numpy()
                confs = pred["conf"].cpu().numpy()
                cls_ids = pred["cls"].cpu().numpy().astype(int)

                # Scale 2D bboxes from network input space to original image space
                bboxes_orig = bboxes.copy()
                bboxes_orig[:, 0] /= ratio_w
                bboxes_orig[:, 2] /= ratio_w
                bboxes_orig[:, 1] /= ratio_h
                bboxes_orig[:, 3] /= ratio_h

                for j in range(bboxes_orig.shape[0]):
                    score = float(confs[j])
                    if score < threshold:
                        continue
                    cls_id = cls_ids[j]
                    if cls_id < 0 or cls_id >= len(class_names):
                        continue
                    x1, y1, x2, y2 = map(int, bboxes_orig[j])
                    cv2.rectangle(img_2d, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for pred
                    label = f"{class_names[cls_id]} {score:.2f}"
                    cv2.putText(img_2d, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imwrite(os.path.join(output_dir, f'id_{img_id:06d}_2d.png'), img_2d)

                # --- 3D Box Visualization ---
                img_3d = img_base.copy()

                # Draw GT 3D boxes
                for obj in gt_objects:
                    if obj.cls_type == 'DontCare':
                        continue
                    if obj.cls_type not in class_names:
                        continue
                    try:
                        corners3d = obj.generate_corners3d()
                        pts_2d, _ = calib.rect_to_img(corners3d)
                        img_3d = _draw_projected_box3d(img_3d, pts_2d, color=(255, 0, 0))  # Blue GT
                    except Exception:
                        continue

                # Draw predicted 3D boxes
                mono3d = pred.get("mono3d", None)
                if mono3d is not None and mono3d.numel() > 0:
                    mono3d_np = mono3d.cpu().numpy()
                    for j in range(bboxes.shape[0]):
                        score = float(confs[j])
                        if score < threshold:
                            continue
                        cls_id = cls_ids[j]
                        if cls_id < 0 or cls_id >= len(class_names):
                            continue

                        depth = float(mono3d_np[j, 0])
                        x3d_net = float(mono3d_np[j, 2])
                        y3d_net = float(mono3d_np[j, 3])
                        size_3d_res = mono3d_np[j, 4:7]
                        heading_raw = mono3d_np[j, 7:31]

                        # Convert 3D center from network input space to original image space
                        x3d = x3d_net / ratio_w
                        y3d = y3d_net / ratio_h

                        dimensions = size_3d_res + cls_mean_size[cls_id]
                        h, w, l = dimensions

                        heading_bin = heading_raw[:self.num_heading_bin]
                        heading_res = heading_raw[self.num_heading_bin:]
                        cls_bin = int(np.argmax(heading_bin))
                        res_val = float(heading_res[cls_bin])
                        alpha = class2angle(cls_bin, res_val, to_label_format=True)

                        locations = calib.img_to_rect(
                            np.array([x3d]), np.array([y3d]), np.array([depth])
                        ).reshape(-1)
                        locations[1] += h / 2

                        ry = calib.alpha2ry(alpha, x3d)

                        # Generate 3D box corners
                        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
                        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
                        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
                        R = np.array([[np.cos(ry), 0, np.sin(ry)],
                                      [0, 1, 0],
                                      [-np.sin(ry), 0, np.cos(ry)]])
                        corners3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners])).T + locations

                        pts_2d, _ = calib.rect_to_img(corners3d)
                        img_3d = _draw_projected_box3d(img_3d, pts_2d, color=(0, 255, 0))  # Green pred

                cv2.imwrite(os.path.join(output_dir, f'id_{img_id:06d}_3d.png'), img_3d)

    def save_3d_results(self, preds_list, batch, output_dir):
        """
        Decode 3D predictions and save in KITTI format for official evaluation.

        Args:
            preds_list: list of prediction dicts (from postprocess)
            batch: the batch dict
            output_dir: directory to save result txt files
        """
        os.makedirs(output_dir, exist_ok=True)
        dataset = self.dataloader.dataset

        class_names = dataset.class_name
        cls_mean_size = dataset.cls_mean_size

        img_ids = batch.get("img_id", [])
        ratio_pads = batch.get("ratio_pad", None)

        for i, pred in enumerate(preds_list):
            if i >= len(img_ids):
                break

            img_id = int(img_ids[i]) if not isinstance(img_ids[i], int) else img_ids[i]
            calib = dataset.get_calib(img_id)

            # Get scale ratio: network input space -> original image space
            if ratio_pads is not None and i < len(ratio_pads):
                ratio_w, ratio_h = ratio_pads[i][0]
            else:
                ratio_w, ratio_h = 1.0, 1.0

            output_path = os.path.join(output_dir, '%06d.txt' % img_id)
            lines = []

            bboxes = pred["bboxes"].cpu().numpy()     # [N, 4] xyxy
            confs = pred["conf"].cpu().numpy()         # [N]
            cls_ids = pred["cls"].cpu().numpy().astype(int)  # [N]
            mono3d = pred.get("mono3d", None)

            if mono3d is None or mono3d.numel() == 0 or bboxes.shape[0] == 0:
                with open(output_path, 'w') as f:
                    f.write('')
                continue

            mono3d = mono3d.cpu().numpy()  # [N, 31]

            for j in range(bboxes.shape[0]):
                cls_id = cls_ids[j]
                if cls_id < 0 or cls_id >= len(class_names):
                    continue

                class_name = class_names[cls_id]
                score = float(confs[j])

                # 2D bbox — scale from network input space to original image space
                x1, y1, x2, y2 = bboxes[j]
                x1 /= ratio_w
                x2 /= ratio_w
                y1 /= ratio_h
                y2 /= ratio_h

                # 3D predictions
                depth = float(mono3d[j, 0])
                # sigma = float(mono3d[j, 1])  # uncertainty, not saved
                x3d_net = float(mono3d[j, 2])
                y3d_net = float(mono3d[j, 3])
                # Convert 3D center from network input space to original image space
                x3d = x3d_net / ratio_w
                y3d = y3d_net / ratio_h
                size_3d_res = mono3d[j, 4:7]    # residual
                heading_raw = mono3d[j, 7:31]   # 24 values

                # Decode 3D size
                dimensions = size_3d_res + cls_mean_size[cls_id]  # H, W, L
                h3d, w3d, l3d = dimensions

                # Decode heading angle
                heading_bin = heading_raw[:self.num_heading_bin]
                heading_res = heading_raw[self.num_heading_bin:]
                cls_bin = int(np.argmax(heading_bin))
                res_val = float(heading_res[cls_bin])
                alpha = class2angle(cls_bin, res_val, to_label_format=True)

                # 3D location from projected center + depth
                locations = calib.img_to_rect(
                    np.array([x3d]), np.array([y3d]), np.array([depth])
                ).reshape(-1)
                locations[1] += h3d / 2  # center to bottom

                ry = calib.alpha2ry(alpha, x3d)

                # KITTI format line (coordinates now in original image space)
                line = (f"{class_name} 0.0 0 {alpha:.2f} "
                        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                        f"{h3d:.2f} {w3d:.2f} {l3d:.2f} "
                        f"{locations[0]:.2f} {locations[1]:.2f} {locations[2]:.2f} "
                        f"{ry:.2f} {score:.2f}")
                lines.append(line)

            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))

    def _ensure_all_kitti_files(self):
        """
        Ensure all images in the val set have a corresponding result file.
        Some images may have no detections, but KITTI eval requires all files.
        """
        if self._kitti_results_dir is None or self.dataloader is None:
            return
        dataset = self.dataloader.dataset
        for img_id_str in dataset.idx_list:
            img_id = int(img_id_str)
            output_path = os.path.join(self._kitti_results_dir, '%06d.txt' % img_id)
            if not os.path.exists(output_path):
                with open(output_path, 'w') as f:
                    f.write('')

    def get_stats(self):
        """Get stats and also ensure all KITTI result files exist."""
        if RANK in {-1, 0}:
            self._ensure_all_kitti_files()
        stats = super().get_stats()
        # Inject KITTI 3D metrics into stats if available
        if self._kitti_3d_metrics:
            stats.update(self._kitti_3d_metrics)
        return stats

    @property
    def kitti_3d_metrics(self):
        """Access KITTI 3D evaluation metrics."""
        return self._kitti_3d_metrics

    def get_desc(self):
        """Return a formatted description string for progress display."""
        return ("%22s" + "%11s" * 6) % (
            "Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)",
        )


class _UltralyticsLoggerAdapter:
    """Adapter to make LOGGER work with MonoDDLE's eval() logger interface."""

    def info(self, msg):
        LOGGER.info(msg)

    def warning(self, msg):
        LOGGER.warning(msg)

    def success(self, msg):
        LOGGER.info(f"[SUCCESS] {msg}")

    def print_section(self, *args, **kwargs):
        pass


def _draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """Draw 3D bounding box projected onto 2D image."""
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image
