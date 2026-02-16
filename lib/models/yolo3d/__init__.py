"""
YOLO3D: Monocular 3D Object Detection based on Ultralytics YOLO.

Uses YOLO's 2D detection pipeline (backbone + neck + detection head) and extends it
with monocular 3D prediction branches (depth, 3D dimensions, heading, 3D center offset).
Reuses Ultralytics' training loop, data augmentation, and post-processing.

Supported YOLO versions:
  - YOLOv8 (yolov8n/s/m/l/x) — standard NMS, reg_max=16
  - YOLO11 (yolo11n/s/m/l/x)  — improved backbone, standard NMS
  - YOLO26 (yolo26n/s/m/l/x)  — NMS-free end2end, reg_max=1

Architecture follows the Pose extension pattern:
  Detect3D(Detect) -> Detection3DModel(DetectionModel) -> Detection3DTrainer(DetectionTrainer)

For YOLO26 end2end models:
  - Detect3D creates dual one2many/one2one heads (including cv4 for 3D)
  - E2ELoss wraps v8Detection3DLoss with decaying weight schedule
  - postprocess() uses topk instead of NMS, preserving 3D channels
"""

from lib.models.yolo3d.head import Detect3D
from lib.models.yolo3d.model import Detection3DModel, YOLO3D
from lib.models.yolo3d.loss import v8Detection3DLoss
from lib.models.yolo3d.train import Detection3DTrainer
from lib.models.yolo3d.val import Detection3DValidator

__all__ = [
    "Detect3D",
    "Detection3DModel",
    "YOLO3D",
    "v8Detection3DLoss",
    "Detection3DTrainer",
    "Detection3DValidator",
]
