import torch.nn as nn
from timm.scheduler import MultiStepLRScheduler


def build_lr_scheduler(cfg, optimizer, last_epoch):
    decay_list = cfg.get('decay_list', [])
    decay_rate = cfg.get('decay_rate', 0.1)
    
    # Warmup config
    warmup_epochs = 5 if cfg.get('warmup', False) else 0
    warmup_lr = 1e-5

    lr_scheduler = MultiStepLRScheduler(
        optimizer,
        decay_t=decay_list,
        decay_rate=decay_rate,
        warmup_t=warmup_epochs,
        warmup_lr_init=warmup_lr,
        t_in_epochs=True
    )

    if last_epoch != -1:
        lr_scheduler.step(last_epoch + 1)
        
    return lr_scheduler, None


def build_bnm_scheduler(cfg, model, last_epoch):
    if not cfg['enabled']:
        return None

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['decay_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['decay_rate']
        return max(cfg['momentum']*cur_decay, cfg['clip'])

    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return bnm_scheduler


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))



