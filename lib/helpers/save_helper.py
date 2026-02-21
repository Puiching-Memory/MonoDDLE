import os
import re
import torch
import torch.nn as nn


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, (torch.nn.DataParallel,
                              torch.nn.parallel.DistributedDataParallel)):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, map_location, logger=None):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch


def cleanup_old_checkpoints(ckpt_dir, max_checkpoints=5, logger=None):
    """Remove oldest checkpoints to keep at most ``max_checkpoints`` files.

    Parameters
    ----------
    ckpt_dir : str
        Directory containing checkpoint files.
    max_checkpoints : int
        Maximum number of checkpoint files to retain. ``0`` disables cleanup.
    logger : optional
        Logger instance for informational messages.
    """
    if max_checkpoints <= 0:
        return

    pattern = re.compile(r'^checkpoint_epoch_(\d+)\.pth$')
    ckpt_files = []
    for fname in os.listdir(ckpt_dir):
        match = pattern.match(fname)
        if match:
            epoch_num = int(match.group(1))
            ckpt_files.append((epoch_num, fname))

    ckpt_files.sort(key=lambda x: x[0])

    if len(ckpt_files) <= max_checkpoints:
        return

    to_remove = ckpt_files[:len(ckpt_files) - max_checkpoints]
    for epoch_num, fname in to_remove:
        fpath = os.path.join(ckpt_dir, fname)
        try:
            os.remove(fpath)
            if logger:
                logger.info("Removed old checkpoint: %s", fname)
        except OSError as e:
            if logger:
                logger.warning("Failed to remove checkpoint %s: %s", fname, e)