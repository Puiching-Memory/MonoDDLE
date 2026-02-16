"""
Detect3D: YOLO detection head extended with monocular 3D prediction branches.

Compatible with YOLOv8, YOLO11, and YOLO26 (including NMS-free end2end).

Following the Pose extension pattern, adds a `cv4` head for 3D parameters:
  - depth (2): raw depth + log variance for uncertainty
  - offset_3d (2): sub-pixel offset of projected 3D center from anchor point
  - size_3d (3): height, width, length residuals from class mean size
  - heading (2*num_heading_bin): bin classification + bin residual for heading angle

Total: 2+2+3+2*num_heading_bin extra channels per anchor point (default 31).

For YOLO26 end2end models, one2one_cv4 is created alongside one2one_cv2/cv3,
and postprocess() is overridden to preserve 3D channels through topk selection.
"""

import copy
import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv


class Detect3D(Detect):
    """YOLO Detect head extended with monocular 3D estimation branches.

    Works with all YOLO versions that use the standard Detect head:
      - YOLOv8 (reg_max=16, end2end=False)
      - YOLO11  (reg_max=16, end2end=False)
      - YOLO26  (reg_max=1,  end2end=True — NMS-free)
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=(), num_heading_bin=12):
        """
        Args:
            nc: Number of classes.
            reg_max: DFL regression max value (16 for v8/11, 1 for v26).
            end2end: Whether to use end2end NMS-free detection (True for v26).
            ch: Tuple of input channel sizes from each detection scale.
            num_heading_bin: Number of heading angle bins (default: 12).
        """
        super().__init__(nc, reg_max, end2end, ch)

        self.num_heading_bin = num_heading_bin
        # 3D output: depth(2) + offset_3d(2) + size_3d(3) + heading(2 * num_heading_bin)
        self.n3d = 2 + 2 + 3 + 2 * num_heading_bin

        c4 = max(ch[0] // 4, self.n3d)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.n3d, 1)
            )
            for x in ch
        )

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Return head modules for one-to-many matching branch."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mono3d_head=self.cv4)

    @property
    def one2one(self):
        """Return head modules for one-to-one matching branch (end2end)."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mono3d_head=self.one2one_cv4,
        )

    def forward_head(self, x, box_head=None, cls_head=None, mono3d_head=None):
        """
        Forward pass through detection + 3D heads.

        Args:
            x: List of feature maps [P3, P4, P5].
            box_head: Box regression head modules (cv2).
            cls_head: Classification head modules (cv3).
            mono3d_head: 3D prediction head modules (cv4).

        Returns:
            dict: {boxes, scores, feats, mono3d}
        """
        preds = super().forward_head(x, box_head, cls_head)
        if mono3d_head is not None:
            bs = x[0].shape[0]
            preds["mono3d"] = torch.cat(
                [mono3d_head[i](x[i]).view(bs, self.n3d, -1) for i in range(self.nl)],
                dim=2,
            )
        return preds

    def _inference(self, preds):
        """Decode predictions during inference, appending decoded 3D params."""
        det_out = super()._inference(preds)  # [B, 4+nc, N]

        if "mono3d" in preds:
            # det_out[:, 0:4, :] contains decoded bboxes (xyxy in pixel coords)
            # Compute 2D bbox centers to decode 3D projected center
            bbox_cx = (det_out[:, 0:1, :] + det_out[:, 2:3, :]) / 2.0
            bbox_cy = (det_out[:, 1:2, :] + det_out[:, 3:4, :]) / 2.0
            mono3d_decoded = self._decode_mono3d(preds["mono3d"], bbox_cx, bbox_cy)
            det_out = torch.cat([det_out, mono3d_decoded], dim=1)

        return det_out

    def postprocess(self, preds):
        """
        Post-process for end2end (NMS-free) models.

        Override Detect.postprocess to correctly handle 3D channels.
        Standard Detect.postprocess splits [4, nc] which would fail with
        extra 3D channels. We split [4, nc, n3d] and re-gather all three.

        Args:
            preds: [B, N, 4+nc+n3d] predictions from _inference().permute(0,2,1)

        Returns:
            [B, min(max_det, N), 6+n3d]: [x,y,w,h, max_score, cls_id, 3d_channels...]
        """
        n_extra = preds.shape[-1] - 4 - self.nc
        if n_extra <= 0:
            # No 3D channels, fall back to standard postprocess
            return super().postprocess(preds)

        boxes, scores, extra = preds.split([4, self.nc, n_extra], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        extra = extra.gather(dim=1, index=idx.repeat(1, 1, n_extra))
        return torch.cat([boxes, scores, conf, extra], dim=-1)

    def fuse(self):
        """Remove one2many heads for efficient end2end inference."""
        if self.end2end:
            self.cv2 = self.cv3 = self.cv4 = None

    def _decode_mono3d(self, mono3d, bbox_cx, bbox_cy):
        """
        Decode raw 3D predictions to interpretable values.

        Input mono3d: [B, n3d, N] where N = total anchor points across scales.
        Input bbox_cx, bbox_cy: [B, 1, N] decoded 2D bbox centers in pixel coords.
        Output: [B, n3d_decoded, N] with:
          - depth (1): decoded metric depth
          - sigma (1): uncertainty weight (exp(-log_var))
          - x3d (1): projected 3D center x in pixel coords
          - y3d (1): projected 3D center y in pixel coords
          - size_3d (3): raw residual (to be added to class mean)
          - heading (2*nhb): raw heading bins + residuals
        Total: 31 channels (when num_heading_bin=12)
        """
        nhb = self.num_heading_bin

        raw_depth = mono3d[:, 0:1, :]
        log_var = mono3d[:, 1:2, :]
        offset_3d = mono3d[:, 2:4, :]
        size_3d = mono3d[:, 4:7, :]
        heading = mono3d[:, 7 : 7 + 2 * nhb, :]

        # Decode depth: inverse sigmoid mapping (MonoDDLE style)
        depth = 1.0 / (raw_depth.sigmoid() + 1e-6) - 1.0
        sigma = torch.exp(-log_var)

        # Decode 3D center: 2D bbox center + raw offset (in pixel coords)
        # The network is trained with L1 loss: offset_3d_target = center_3d_img - center_2d
        # So: center_3d_img = center_2d + offset_3d_pred
        x3d = bbox_cx + offset_3d[:, 0:1, :]
        y3d = bbox_cy + offset_3d[:, 1:2, :]

        return torch.cat([depth, sigma, x3d, y3d, size_3d, heading], dim=1)
