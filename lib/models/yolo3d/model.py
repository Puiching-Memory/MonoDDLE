"""
Detection3DModel: Ultralytics DetectionModel extended for monocular 3D detection.

Supports multiple YOLO architectures via head-swapping:
  - YOLOv8  (yolov8n/s/m/l/x.yaml)  — standard NMS
  - YOLO11  (yolo11n/s/m/l/x.yaml)  — standard NMS
  - YOLO26  (yolo26n/s/m/l/x.yaml)  — NMS-free end2end

Strategy:
  Since ultralytics' parse_model has a hardcoded frozenset of head types,
  we build a standard DetectionModel (with Detect head) and then surgically
  replace the Detect head with our Detect3D, preserving the pretrained 2D
  detection weights (cv2, cv3, dfl) and adding the new 3D branch (cv4).

  For YOLO26 end2end models, one2one head weights (one2one_cv2/cv3) are also
  transferred, and E2ELoss is used instead of v8Detection3DLoss.
"""

import numpy as np

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.head import Detect
from ultralytics.engine.model import Model
from ultralytics.utils import LOGGER

from lib.models.yolo3d.head import Detect3D


class Detection3DModel(DetectionModel):
    """Ultralytics DetectionModel with an additional monocular 3D detection head.

    Supports:
      - YOLOv8 (reg_max=16, end2end=False): standard anchor-free with NMS
      - YOLO11 (reg_max=16, end2end=False): improved backbone, same detection
      - YOLO26 (reg_max=1,  end2end=True):  NMS-free end-to-end detection
    """

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, num_heading_bin=12,
                 cls_mean_size=None, verbose=True):
        """
        Build a standard DetectionModel from cfg, then swap Detect → Detect3D.

        Args:
            cfg: Standard YOLO model YAML path or dict.
                 Supports: yolov8*.yaml, yolo11*.yaml, yolo26*.yaml
            ch: Input channels.
            nc: Number of classes.
            num_heading_bin: Number of heading angle bins for 3D orientation.
            cls_mean_size: np.ndarray (num_class, 3) mean H/W/L per class.
            verbose: Print model summary.
        """
        # Build a standard detection model (backbone + neck + Detect head)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # --- Swap the Detect head with Detect3D ---
        m = self.model[-1]
        assert isinstance(m, Detect), f"Expected last module to be Detect, got {type(m)}"

        # Extract per-scale input channels from existing Detect head
        ch_list = tuple(cv2_seq[0].conv.in_channels for cv2_seq in m.cv2)
        end2end = m.end2end  # True for YOLO26, False for v8/v11

        # Create Detect3D: reuses 2D head structure, adds 3D branch (cv4)
        detect3d = Detect3D(
            nc=m.nc,
            reg_max=m.reg_max,
            end2end=end2end,
            ch=ch_list,
            num_heading_bin=num_heading_bin,
        )

        # ---- Transfer 2D detection weights from Detect → Detect3D ----
        detect3d.cv2 = m.cv2
        detect3d.cv3 = m.cv3
        detect3d.dfl = m.dfl
        detect3d.stride = m.stride

        # Transfer one2one heads for end2end models (YOLO26)
        if end2end and hasattr(m, "one2one_cv2"):
            detect3d.one2one_cv2 = m.one2one_cv2
            detect3d.one2one_cv3 = m.one2one_cv3

        # Copy bookkeeping attributes used by the framework
        detect3d.i = m.i
        detect3d.f = m.f
        detect3d.type = type(detect3d).__name__
        detect3d.inplace = getattr(m, "inplace", True)

        self.model[-1] = detect3d
        self.stride = detect3d.stride

        # Store 3D metadata
        self.num_heading_bin = num_heading_bin
        if cls_mean_size is not None:
            self.cls_mean_size = np.array(cls_mean_size, dtype=np.float32)
        else:
            self.cls_mean_size = np.zeros((nc or 3, 3), dtype=np.float32)

        if verbose:
            e2e_str = " [end2end/NMS-free]" if end2end else ""
            LOGGER.info(f"Detect3D head installed{e2e_str}: n3d={detect3d.n3d}, "
                        f"num_heading_bin={num_heading_bin}, reg_max={m.reg_max}, ch={ch_list}")

    def init_criterion(self):
        """Initialize the monocular 3D detection loss.

        For end2end models (YOLO26), wraps v8Detection3DLoss in E2ELoss
        with dual one2many/one2one branches and decaying weight schedule.
        For standard models (v8/v11), uses v8Detection3DLoss directly.
        """
        from lib.models.yolo3d.loss import v8Detection3DLoss

        if getattr(self, "end2end", False):
            from ultralytics.utils.loss import E2ELoss
            return E2ELoss(self, v8Detection3DLoss)
        return v8Detection3DLoss(self)


class YOLO3D(Model):
    """
    YOLO3D model for monocular 3D object detection.

    Builds on ultralytics YOLO with an added 3D detection branch.
    Supports all YOLO versions: v8, 11, 26 (including NMS-free).

    Usage:
        # YOLOv8 — standard NMS
        model = YOLO3D("yolov8n.pt")
        model = YOLO3D("yolov8n.yaml")

        # YOLO11 — standard NMS
        model = YOLO3D("yolo11n.pt")
        model = YOLO3D("yolo11n.yaml")

        # YOLO26 — NMS-free end2end
        model = YOLO3D("yolo26n.pt")
        model = YOLO3D("yolo26n.yaml")

        # Train
        model.train(data="kitti3d.yaml", epochs=100)
    """

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        super().__init__(model=model, task=task or "detect3d", verbose=verbose)
        # Force task to "detect3d" — loading a pretrained .pt will override
        # self.task to "detect" (the task stored in the checkpoint), which
        # causes _smart_load to fail because task_map only has "detect3d".
        self.task = "detect3d"

    @property
    def task_map(self):
        from lib.models.yolo3d.train import Detection3DTrainer
        from lib.models.yolo3d.val import Detection3DValidator
        from ultralytics.models.yolo.detect.predict import DetectionPredictor

        return {
            "detect3d": {
                "model": Detection3DModel,
                "trainer": Detection3DTrainer,
                "validator": Detection3DValidator,
                "predictor": DetectionPredictor,
            }
        }
