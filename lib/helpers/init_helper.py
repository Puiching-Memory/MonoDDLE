import math
import torch
import torch.nn as nn

def init_weights_kaiming(m):
    """
    Kaiming initialization for Conv2d and Linear layers.
    Better for layers followed by ReLU.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Kaiming Normal is generally preferred over Uniform for ReLU networks
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def init_weights_normal(m, std=0.001):
    """
    Normal initialization with small std.
    Used for head layers in CenterNet.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_head_bias(m, prior_prob=0.1):
    """
    Initialize bias for the final layer of a focal loss head (e.g., heatmap).
    bias = -log((1 - prior_prob) / prior_prob)
    """
    bias_val = -math.log((1 - prior_prob) / prior_prob)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, bias_val)

def fill_up_weights(up):
    """
    Bilinear interpolation initialization for ConvTranspose2d (upsampling).
    """
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]
