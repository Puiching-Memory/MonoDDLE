"""
Checkpoint save / load utilities.

Supports DataParallel, DistributedDataParallel, EMA, AMP GradScaler,
and torch.compile wrapped models.
"""

import os
import torch
import torch.nn as nn


def _unwrap_model(model):
    """Unwrap DataParallel / DDP / compiled model to get the raw module."""
    m = model
    # torch.compile wraps in OptimizedModule (torch >= 2.0)
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        m = m.module
    return m


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None,
                         scaler=None, ema=None, extra=None):
    """Build a checkpoint dict.

    Args:
        model: the training model (may be DP/DDP/compiled wrapped).
        optimizer: optimizer (optional).
        epoch: current epoch number.
        scaler: ``torch.amp.GradScaler`` instance (optional).
        ema: ``ModelEMA`` instance (optional).
        extra: dict of arbitrary extra state to persist.
    """
    optim_state = optimizer.state_dict() if optimizer is not None else None

    if model is not None:
        raw = _unwrap_model(model)
        model_state = model_state_to_cpu(raw.state_dict())
    else:
        model_state = None

    state = {
        'epoch': epoch,
        'model_state': model_state,
        'optimizer_state': optim_state,
    }

    if scaler is not None:
        state['scaler_state'] = scaler.state_dict()

    if ema is not None:
        state['ema_state'] = ema.state_dict()

    if extra is not None:
        state.update(extra)

    return state


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, map_location,
                    logger=None, scaler=None, ema=None):
    """Load model / optimizer / scaler / ema from a checkpoint file.

    Returns:
        epoch saved in the checkpoint (int).
    """
    if os.path.isfile(filename):
        if logger:
            logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=map_location,
                                weights_only=False)
        epoch = checkpoint.get('epoch', -1)

        if model is not None and checkpoint.get('model_state') is not None:
            raw = _unwrap_model(model)
            raw.load_state_dict(checkpoint['model_state'], strict=False)

        if optimizer is not None and checkpoint.get('optimizer_state') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        if scaler is not None and checkpoint.get('scaler_state') is not None:
            scaler.load_state_dict(checkpoint['scaler_state'])

        if ema is not None and checkpoint.get('ema_state') is not None:
            ema.load_state_dict(checkpoint['ema_state'])

        if logger:
            logger.info("==> Done (epoch %d)" % epoch)
    else:
        raise FileNotFoundError("Checkpoint not found: %s" % filename)

    return epoch