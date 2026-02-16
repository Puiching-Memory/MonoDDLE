"""
CenterNet3D with timm backbone for monocular 3D object detection.

Uses ``timm.create_model(..., features_only=True)`` to extract multi-scale
feature maps from *any* timm-registered backbone (ResNet, EfficientNet,
ConvNeXt, Swin, etc.) and feeds them into the same DLAUp neck + detection
heads as the original CenterNet3D.
"""

import numpy as np
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

from lib.necks.dlaup import DLAUp


class CenterNet3DTimm(nn.Module):
    """CenterNet3D with a pluggable timm backbone.

    Args:
        backbone_name: any model name recognised by ``timm.create_model``.
        pretrained: load ImageNet pretrained weights.
        neck: feature fusion module (currently only 'DLAUp').
        num_class: number of object classes.
        downsample: output stride (4 / 8 / 16 / 32).
    """

    def __init__(self, backbone_name='resnet50', pretrained=True,
                 neck='DLAUp', num_class=3, downsample=4):
        assert downsample in [4, 8, 16, 32]
        super().__init__()

        if timm is None:
            raise ImportError("timm is required for CenterNet3DTimm")

        self.heads = {
            'heatmap': num_class,
            'offset_2d': 2,
            'size_2d': 2,
            'depth': 2,
            'offset_3d': 2,
            'size_3d': 3,
            'heading': 24,
        }

        # Create timm backbone with multi-scale feature extraction
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )
        channels = self.backbone.feature_info.channels()  # e.g. [64, 128, 256, 512] for resnet
        # Store channels for compatibility
        self.channels = channels

        self.first_level = int(np.log2(downsample)) - 1  # timm levels are 0-indexed from stride 2
        # Ensure first_level is valid
        self.first_level = max(0, min(self.first_level, len(channels) - 1))

        used_channels = channels[self.first_level:]
        scales = [2 ** i for i in range(len(used_channels))]
        self.neck_module = DLAUp(used_channels, scales_list=scales)

        # Build detection heads
        head_in_channels = used_channels[0]
        for head in self.heads:
            output_channels = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(head_in_channels, 256, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, output_channels, kernel_size=1, stride=1, padding=0, bias=True),
            )
            if 'heatmap' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self._fill_fc_weights(fc)
            self.__setattr__(head, fc)

    @staticmethod
    def _fill_fc_weights(layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = self.backbone(x)  # list of multi-scale features
        feat = self.neck_module(feats[self.first_level:])
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)
        return ret
