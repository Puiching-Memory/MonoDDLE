import math
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

def build_optimizer(cfg_optimizer, model):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]

    parameters = [{'params': biases, 'weight_decay': 0},
                  {'params': weights, 'weight_decay': cfg_optimizer['weight_decay']}]


    if cfg_optimizer['type'] == 'sgd':
        optimizer = optim.SGD(parameters, lr=cfg_optimizer['lr'], momentum=0.9)
    elif cfg_optimizer['type'] == 'adam':
        optimizer = optim.Adam(parameters, lr=cfg_optimizer['lr'])
    elif cfg_optimizer['type'] == 'adamw':
        optimizer = optim.AdamW(parameters, lr=cfg_optimizer['lr'])
    else:
        raise NotImplementedError("%s optimizer is not supported" % cfg_optimizer['type'])

    return optimizer