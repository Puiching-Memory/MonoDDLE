"""
Model builder.

Supports:
  - ``centernet3d``  — original MonoDLE CenterNet3D (DLA backbone)
  - ``centernet3d_timm`` — CenterNet3D with *any* timm backbone

Config examples::

    # Original DLA34 backbone
    model:
      type: centernet3d
      backbone: dla34
      neck: DLAUp
      num_class: 3

    # timm backbone (e.g. ResNet-50, EfficientNet, ConvNeXt, …)
    model:
      type: centernet3d_timm
      backbone: resnet50            # any timm model name
      pretrained: true
      neck: DLAUp
      num_class: 3
"""

from lib.models.centernet3d import CenterNet3D


def build_model(cfg):
    model_type = cfg['type']

    if model_type == 'centernet3d':
        return CenterNet3D(
            backbone=cfg['backbone'],
            neck=cfg['neck'],
            num_class=cfg['num_class'],
        )

    if model_type == 'centernet3d_timm':
        return _build_centernet3d_timm(cfg)

    raise NotImplementedError("%s model is not supported" % model_type)


def _build_centernet3d_timm(cfg):
    """Build CenterNet3D with a timm feature-extractor backbone."""
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for centernet3d_timm. "
                          "Install it with: pip install timm")

    from lib.models.centernet3d_timm import CenterNet3DTimm

    return CenterNet3DTimm(
        backbone_name=cfg['backbone'],
        pretrained=cfg.get('pretrained', True),
        neck=cfg.get('neck', 'DLAUp'),
        num_class=cfg.get('num_class', 3),
        downsample=cfg.get('downsample', 4),
    )