from lib.models.centernet3d import CenterNet3D
import torch


def log_model_complexity(model, logger=None, input_resolution=(384, 1280), device=None):
    try:
        from thop import profile, clever_format
    except ImportError:
        msg = 'ultralytics-thop is not installed, skip FLOPs/Params profiling.'
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)
        return None

    h, w = int(input_resolution[0]), int(input_resolution[1])
    if device is None:
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device('cpu')

    prev_mode = model.training
    model.eval()
    dummy_input = torch.randn(1, 3, h, w, device=device)

    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # thop may attach temporary attributes/buffers (e.g. total_ops/total_params)
    # onto modules during profiling. Remove them to avoid state mismatch between
    # ranks when only rank-0 runs complexity profiling before DDP wrapping.
    for module in model.modules():
        if hasattr(module, 'total_ops'):
            delattr(module, 'total_ops')
        if hasattr(module, 'total_params'):
            delattr(module, 'total_params')

    if prev_mode:
        model.train()

    flops_str, params_str = clever_format([flops, params], '%.3f')
    msg = f'Model Complexity | Input: (1,3,{h},{w}) | FLOPs: {flops_str} | Params: {params_str}'
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

    return {'flops': flops, 'params': params}


def build_model(cfg):
    if cfg['type'] == 'centernet3d':
        return CenterNet3D(
            backbone=cfg['backbone'],
            neck=cfg['neck'],
            num_class=cfg['num_class'],
            downsample=cfg.get('downsample', 4),
            model_cfg=cfg,
        )
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])


