import os
import torch
import torch.nn as nn
import numpy as np
import timm

from lib.necks.dlaup import DLAUp
from lib.helpers.init_helper import init_weights_normal, init_head_bias

class CenterNet3D(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', num_class=3, downsample=4):
        """
        CenterNet for monocular 3D object detection.
        :param backbone: the backbone of pipeline, such as dla34.
        :param neck: the necks of detection, such as dla_up.
        :param downsample: the ratio of down sample. [4, 8, 16, 32]
        :param head_conv: the channels of convolution in head. default: 256
        """
        assert downsample in [4, 8, 16, 32]
        super().__init__()

        self.heads = {'heatmap': num_class, 'offset_2d': 2, 'size_2d' :2, 'depth': 2, 'offset_3d': 2, 'size_3d':3, 'heading': 24}

        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        channels = self.backbone.feature_info.channels()
        reductions = self.backbone.feature_info.reduction()
        if downsample not in reductions:
            raise ValueError('downsample=%s not in backbone reductions %s' % (downsample, reductions))
        self.first_level = reductions.index(downsample)
        base = reductions[self.first_level]
        scales = [r // base for r in reductions[self.first_level:]]
        self.neck = DLAUp(channels[self.first_level:], scales_list=scales)   # feature fusion [such as DLAup, FPN]

        # initialize the head of pipeline, according to heads setting.
        for head in self.heads.keys():
            output_channels = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(channels[self.first_level], 256, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, output_channels, kernel_size=1, stride=1, padding=0, bias=True))

            # initialization
            fc.apply(init_weights_normal)
            if 'heatmap' in head:
                init_head_bias(fc[-1], prior_prob=0.1)

            self.__setattr__(head, fc)


    def forward(self, input):
        feat = self.backbone(input)
        feat = self.neck(feat[self.first_level:])

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)

        return ret




if __name__ == '__main__':
    import torch
    net = CenterNet3D(backbone='dla34')
    print(net)

    input = torch.randn(4, 3, 384, 1280)
    print(input.shape, input.dtype)
    output = net(input)
    print(output.keys())


