import os
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.backbones import dla
from lib.backbones.dlaup import DLAUp
from lib.backbones.hourglass import get_large_hourglass_net
from lib.backbones.hourglass import load_pretrian_model
from lib.backbones.ultralytics_adapter import UltralyticsBackboneAdapter


class NativeNeck(nn.Module):
    """Top-down FPN that fuses P3/P4/P5 neck features into a stride-4 map."""

    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.laterals = nn.ModuleList(
            [nn.Conv2d(ch, out_channels, 1, bias=False) for ch in in_channels_list]
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample_to_stride4 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.ConvTranspose2d):
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2.0 * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
                for ch in range(1, w.size(0)):
                    w[ch, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, features):
        """features: list of [P3(stride8), P4(stride16), P5(stride32)]."""
        lats = [lat(f) for lat, f in zip(self.laterals, features)]
        for i in range(len(lats) - 2, -1, -1):
            lats[i] = lats[i] + F.interpolate(
                lats[i + 1], size=lats[i].shape[2:], mode="nearest"
            )
        return self.upsample_to_stride4(self.smooth(lats[0]))


class CenterNet3D(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', num_class=3, downsample=4, model_cfg=None):
        """
        CenterNet for monocular 3D object detection.
        :param backbone: the backbone of pipeline, such as dla34.
        :param neck: 'DLAUp' or 'NativeNeck'. NativeNeck uses YOLO PANet output directly.
        :param downsample: the ratio of down sample. [4, 8, 16, 32]
        """
        assert downsample in [4, 8, 16, 32]
        super().__init__()
        model_cfg = model_cfg or {}

        self.heads = {'heatmap': num_class, 'offset_2d': 2, 'size_2d' :2, 'depth': 2, 'offset_3d': 2, 'size_3d':3, 'heading': 24}
        if str(backbone).lower().startswith('yolo'):
            default_model = {
                'yolov8': 'yolov8n.pt',
                'yolo8': 'yolov8n.pt',
                'yolo11': 'yolo11n.pt',
                'yolo26': 'yolo26n.pt',
            }.get(str(backbone).lower(), 'yolov8n.pt')
            use_native = (neck == 'NativeNeck')
            self.backbone = UltralyticsBackboneAdapter(
                model_path=model_cfg.get('ultralytics_model', default_model),
                feature_strides=[8, 16, 32] if use_native else model_cfg.get('feature_strides', [4, 8, 16, 32]),
                feature_indices=model_cfg.get('feature_indices', None),
                freeze=model_cfg.get('freeze_backbone', False),
            )
            channels = self.backbone.channels
            self.first_level = 0
            if use_native:
                neck_out = model_cfg.get('neck_out_channels', 64)
                self.neck = NativeNeck(channels, out_channels=neck_out)
                head_in_channels = neck_out
            else:
                scales = [2 ** i for i in range(len(channels))]
                self.neck = DLAUp(channels, scales_list=scales)
                head_in_channels = channels[self.first_level]
        elif str(backbone).lower().startswith('timm_'):
            from lib.backbones.timm_adapter import TimmBackboneAdapter
            model_name = str(backbone)[5:] # remove 'timm_' prefix
            self.backbone = TimmBackboneAdapter(
                model_name=model_name,
                pretrained=model_cfg.get('pretrained', True),
                feature_strides=model_cfg.get('feature_strides', (4, 8, 16, 32)),
                freeze=model_cfg.get('freeze_backbone', False),
            )
            channels = self.backbone.channels
            self.first_level = 0
            scales = [2 ** i for i in range(len(channels))]
            self.neck = DLAUp(channels, scales_list=scales)
            head_in_channels = channels[self.first_level]
        else:
            self.backbone = getattr(dla, backbone)(pretrained=True, return_levels=True)
            channels = self.backbone.channels  # channels list for feature maps generated by backbone
            self.first_level = int(np.log2(downsample))
            scales = [2 ** i for i in range(len(channels[self.first_level:]))]
            self.neck = DLAUp(channels[self.first_level:], scales_list=scales)   # feature fusion [such as DLAup, FPN]
            head_in_channels = channels[self.first_level]

        # initialize the head of pipeline, according to heads setting.
        for head in self.heads.keys():
            output_channels = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(head_in_channels, 256, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, output_channels, kernel_size=1, stride=1, padding=0, bias=True))

            # initialization
            if 'heatmap' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def forward(self, input):
        feat = self.backbone(input)
        if isinstance(self.neck, NativeNeck):
            feat = self.neck(feat)
        else:
            feat = self.neck(feat[self.first_level:])

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)

        return ret


    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




if __name__ == '__main__':
    import torch
    net = CenterNet3D(backbone='dla34')
    print(net)

    input = torch.randn(4, 3, 384, 1280)
    print(input.shape, input.dtype)
    output = net(input)
    print(output.keys())


