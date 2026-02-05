import os
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename):
    # Save model weights using safetensors
    if state['model_state'] is not None:
        model_filename = '{}.safetensors'.format(filename)
        save_file(state['model_state'], model_filename)

    # Save optimizer and other info using torch.save
    optimizer_filename = '{}.optimizer.pth'.format(filename)
    other_state = {k: v for k, v in state.items() if k != 'model_state'}
    torch.save(other_state, optimizer_filename)


def load_checkpoint(model, optimizer, filename, map_location, logger=None):
    if logger:
        logger.info(f"Loading from checkpoint '{filename}'")

    epoch = -1
    
    # Force usage of safetensors
    if not filename.endswith('.safetensors'):
        filename = filename + '.safetensors'

    if os.path.isfile(filename):
        # Load model state
        if model is not None:
            model_state = load_file(filename, device=str(map_location))
            model.load_state_dict(model_state)
        
        # Try to load optimizer state
        optimizer_filename = filename.replace('.safetensors', '.optimizer.pth')
        if os.path.isfile(optimizer_filename):
            checkpoint_opt = torch.load(optimizer_filename, map_location)
            epoch = checkpoint_opt.get('epoch', -1)
            if optimizer is not None and checkpoint_opt.get('optimizer_state') is not None:
                optimizer.load_state_dict(checkpoint_opt['optimizer_state'])
            if logger:
                logger.success(f"Checkpoint (SafeTensors + Opt) loaded successfully (epoch {epoch})")
        else:
            if logger:
                logger.warning(f"Transformation loaded from SafeTensors, but optimizer file '{optimizer_filename}' not found.")
    else:
        raise FileNotFoundError(f"File not found: {filename}")

    return epoch