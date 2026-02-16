"""
v8Detection3DLoss: Extends v8DetectionLoss with monocular 3D losses.

Compatible with both standard and E2E (end2end) training:
  - Standard (v8/v11): Used directly as criterion
  - E2E (v26): Wrapped in E2ELoss with dual one2many/one2one branches

3D loss components (same as MonoDDLE):
  - depth_loss: Laplacian aleatoric uncertainty loss
  - offset3d_loss: L1 loss for projected 3D center offset
  - size3d_loss: Dimension-aware L1 loss for 3D size residuals
  - heading_loss: Cross-entropy (bin) + L1 (residual) for heading angle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy


class v8Detection3DLoss(v8DetectionLoss):
    """YOLOv8 detection loss extended with monocular 3D loss branches.

    Works with:
      - Standard training (v8/v11): loss() called directly by __call__
      - E2E training (v26): loss() called by E2ELoss for each branch
        (one2many with tal_topk=10, one2one with tal_topk2=1)

    E2ELoss compatibility:
      - parse_output(): inherited — extracts preds dict from tuple/dict
      - loss(): returns (total_loss * batch_size, loss_detach) — compatible
      - E2ELoss does: loss_o2m[0]*w_o2m + loss_o2o[0]*w_o2o, loss_o2o[1]
    """

    def __init__(self, model, tal_topk=10, tal_topk2=10):
        super().__init__(model, tal_topk, tal_topk2)
        m = model.model[-1]  # Detect3D module
        self.num_heading_bin = getattr(m, "num_heading_bin", 12)
        self.n3d = getattr(m, "n3d", 31)

        # Loss weights for 3D branches (can be overridden via model.args / hyp)
        self.depth_weight = 1.0
        self.offset3d_weight = 1.0
        self.size3d_weight = 1.0
        self.heading_weight = 1.0

    def loss(self, preds, batch):
        """
        Compute combined 2D detection + 3D monocular loss.

        Args:
            preds: dict with keys 'boxes', 'scores', 'feats', 'mono3d'
                   mono3d shape: [B, n3d, N_total_anchors]
            batch: dict with:
                - Standard ultralytics keys: batch_idx, cls, bboxes, img
                - 3D keys: mono3d_targets (packed 3D annotations)

        Returns:
            (total_loss * batch_size, loss_items_detached)
        """
        batch_size = preds["boxes"].shape[0]

        # --- 2D detection losses (box, cls, dfl) ---
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )

        # --- 3D losses ---
        loss_3d = torch.zeros(4, device=self.device)  # depth, offset3d, size3d, heading

        if fg_mask.sum() > 0 and "mono3d" in preds and "mono3d_targets" in batch:
            pred_mono3d = preds["mono3d"].permute(0, 2, 1).contiguous()  # [B, N, n3d]
            loss_3d = self._compute_3d_losses(
                pred_mono3d, batch, fg_mask, target_gt_idx,
                anchor_points, stride_tensor
            )

        # Combine all losses
        # det_loss[0]=box, det_loss[1]=cls, det_loss[2]=dfl
        loss = torch.zeros(7, device=self.device)
        loss[0] = det_loss[0]  # box
        loss[1] = det_loss[1]  # cls
        loss[2] = det_loss[2]  # dfl
        loss[3] = loss_3d[0] * self.depth_weight      # depth
        loss[4] = loss_3d[1] * self.offset3d_weight    # offset3d
        loss[5] = loss_3d[2] * self.size3d_weight      # size3d
        loss[6] = loss_3d[3] * self.heading_weight     # heading

        return loss.sum() * batch_size, loss.detach()

    def _compute_3d_losses(self, pred_mono3d, batch, fg_mask, target_gt_idx,
                           anchor_points, stride_tensor):
        """
        Compute the 3D loss components for foreground (matched) predictions.

        pred_mono3d: [B, N_anchors, n3d] raw predictions
        batch["mono3d_targets"]: packed tensor from collate, shape [M, 8]
            columns: [depth, offset3d_x, offset3d_y, size3d_h, size3d_w, size3d_l, heading_bin, heading_res]
        batch["batch_idx"]: [M] mapping each target to batch index
        fg_mask: [B, N_anchors] boolean mask of positive assignments
        target_gt_idx: [B, N_anchors] index into GT for each anchor
        """
        device = pred_mono3d.device
        loss = torch.zeros(4, device=device)
        nhb = self.num_heading_bin

        # Recover per-batch targets
        targets_3d = batch["mono3d_targets"].to(device).float()   # [M, 8]
        batch_idx = batch["batch_idx"].to(device).long()
        batch_size = pred_mono3d.shape[0]

        if targets_3d.shape[0] == 0:
            return loss

        # Build batched 3D target tensor: [B, max_gt, 8]
        _, counts = batch_idx.unique(return_counts=True)
        max_gt = counts.max().item()
        batched_targets = torch.zeros(batch_size, max_gt, targets_3d.shape[-1], device=device)
        for i in range(batch_size):
            mask_i = batch_idx == i
            n_i = mask_i.sum()
            if n_i > 0:
                batched_targets[i, :n_i] = targets_3d[mask_i]

        # Select targets for foreground anchors: [B, N_anchors] → gather → [fg_total, 8]
        idx_expanded = target_gt_idx.unsqueeze(-1).expand(-1, -1, targets_3d.shape[-1])
        selected_targets = batched_targets.gather(1, idx_expanded)  # [B, N_anchors, 8]

        # Extract foreground predictions and targets
        fg_pred = pred_mono3d[fg_mask]       # [N_fg, n3d]
        fg_target = selected_targets[fg_mask]  # [N_fg, 8]

        if fg_pred.shape[0] == 0:
            return loss

        # --- Parse predictions ---
        raw_depth = fg_pred[:, 0:1]
        log_var = fg_pred[:, 1:2]
        offset_3d_pred = fg_pred[:, 2:4]
        size_3d_pred = fg_pred[:, 4:7]
        heading_pred = fg_pred[:, 7: 7 + 2 * nhb]

        # --- Parse targets ---
        depth_target = fg_target[:, 0:1]
        offset_3d_target = fg_target[:, 1:3]
        size_3d_target = fg_target[:, 3:6]
        heading_bin_target = fg_target[:, 6].long()
        heading_res_target = fg_target[:, 7]
        mask_3d = fg_target[:, 8] if fg_target.shape[-1] > 8 else torch.ones(fg_target.shape[0], device=device)

        # 1. Depth loss (Laplacian aleatoric uncertainty, same as MonoDDLE)
        depth_decoded = 1.0 / (raw_depth.sigmoid() + 1e-6) - 1.0
        depth_loss = 1.4142 * torch.exp(-log_var) * torch.abs(depth_decoded - depth_target) + log_var
        if mask_3d.sum() > 0:
            loss[0] = (depth_loss.squeeze(-1) * mask_3d).sum() / mask_3d.sum().clamp(min=1)
        else:
            loss[0] = depth_loss.mean()

        # 2. Offset3D loss (L1)
        offset3d_loss = F.l1_loss(offset_3d_pred, offset_3d_target, reduction='none')
        if mask_3d.sum() > 0:
            loss[1] = (offset3d_loss.mean(dim=-1) * mask_3d).sum() / mask_3d.sum().clamp(min=1)
        else:
            loss[1] = offset3d_loss.mean()

        # 3. Size3D loss (dimension-aware L1, same as MonoDDLE)
        loss[2] = self._dim_aware_l1_loss(size_3d_pred, size_3d_target, mask_3d)

        # 4. Heading loss (classification + regression, same as MonoDDLE)
        loss[3] = self._heading_loss(heading_pred, heading_bin_target, heading_res_target, mask_3d)

        return loss

    @staticmethod
    def _dim_aware_l1_loss(pred, target, mask):
        """Dimension-aware L1 loss from MonoDDLE."""
        dimension = target.clone().detach().abs().clamp(min=1e-6)
        raw_loss = torch.abs(pred - target)
        normalized_loss = raw_loss / dimension

        with torch.no_grad():
            compensation = F.l1_loss(pred, target) / normalized_loss.mean().clamp(min=1e-8)
        weighted_loss = normalized_loss * compensation

        if mask.sum() > 0:
            return (weighted_loss.mean(dim=-1) * mask).sum() / mask.sum().clamp(min=1)
        return weighted_loss.mean()

    def _heading_loss(self, heading_pred, heading_bin_target, heading_res_target, mask):
        """
        Heading loss: classification (CE) + regression (L1).

        heading_pred: [N_fg, 2*num_heading_bin] — first half cls, second half res
        heading_bin_target: [N_fg] long
        heading_res_target: [N_fg] float
        mask: [N_fg] float (1.0 for valid 3D)
        """
        nhb = self.num_heading_bin
        heading_cls = heading_pred[:, :nhb]
        heading_res = heading_pred[:, nhb:]

        bool_mask = mask.bool()
        if bool_mask.sum() == 0:
            return torch.tensor(0.0, device=heading_pred.device)

        cls_pred_m = heading_cls[bool_mask]
        cls_target_m = heading_bin_target[bool_mask]
        res_pred_m = heading_res[bool_mask]
        res_target_m = heading_res_target[bool_mask]

        # Classification
        cls_loss = F.cross_entropy(cls_pred_m, cls_target_m, reduction='mean')

        # Regression: select the residual for the target bin
        onehot = torch.zeros_like(res_pred_m).scatter_(
            1, cls_target_m.view(-1, 1), 1.0
        )
        res_selected = (res_pred_m * onehot).sum(dim=1)
        reg_loss = F.l1_loss(res_selected, res_target_m, reduction='mean')

        return cls_loss + reg_loss
