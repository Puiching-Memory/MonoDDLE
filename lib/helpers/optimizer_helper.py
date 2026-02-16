"""
Optimizer builder with support for:
  - PyTorch built-in optimizers (SGD, Adam, AdamW, RAdam, NAdam, etc.)
  - timm optimizers (LAMB, LARS, Adafactor, Lion, etc.)
  - Legacy MonoDLE custom AdamW (kept for backward compat)

Config example (yaml)::

    optimizer:
      type: 'adamw'           # any torch or timm optimizer name
      lr: 0.001
      weight_decay: 0.01
      # --- optional timm-specific fields ---
      timm: false              # force timm optimizer factory
      timm_kwargs:             # extra kwargs forwarded to timm.optim
        eps: 1e-8
"""

import math
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

# timm is optional — graceful fallback
try:
    import timm.optim
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


# ─── Torch built-in optimizer registry ────────────────────────────────
_TORCH_OPTIM = {
    'sgd':     optim.SGD,
    'adam':    optim.Adam,
    'adamw':  optim.AdamW,
    'radam':  optim.RAdam,
    'nadam':  optim.NAdam,
    'adamax': optim.Adamax,
    'rmsprop': optim.RMSprop,
    'rprop':  optim.Rprop,
    'asgd':   optim.ASGD,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
}

# Add LBFGS separately (requires closure)
# _TORCH_OPTIM['lbfgs'] = optim.LBFGS


def _split_params(model, weight_decay):
    """Split parameters into weight-decay and no-weight-decay groups."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't decay bias / norm layers
        if param.ndim <= 1 or 'bias' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def build_optimizer(cfg_optimizer, model):
    """Build an optimizer from a config dict.

    Args:
        cfg_optimizer: dict with at least ``type`` and ``lr``.
        model: nn.Module whose parameters to optimise.

    Returns:
        torch.optim.Optimizer
    """
    opt_type = cfg_optimizer['type'].lower()
    lr = cfg_optimizer['lr']
    wd = cfg_optimizer.get('weight_decay', 0.0)
    params = _split_params(model, wd)

    # ── timm path (explicit or auto-detect) ──────────────────────────
    use_timm = cfg_optimizer.get('timm', False)
    if use_timm or (opt_type not in _TORCH_OPTIM and opt_type != 'custom_adamw'):
        if not _HAS_TIMM:
            raise ImportError(
                f"timm is required for optimizer '{opt_type}'. "
                f"Install it with: pip install timm"
            )
        extra = cfg_optimizer.get('timm_kwargs', {})
        optimizer = timm.optim.create_optimizer_v2(
            model, opt=opt_type, lr=lr, weight_decay=wd, **extra,
        )
        return optimizer

    # ── Legacy custom AdamW (for reproducing original MonoDLE) ───────
    if opt_type == 'custom_adamw':
        return _LegacyAdamW(params, lr=lr)

    # ── Torch built-in ───────────────────────────────────────────────
    cls = _TORCH_OPTIM.get(opt_type)
    if cls is None:
        raise NotImplementedError(f"Optimizer '{opt_type}' is not supported. "
                                  f"Available: {list(_TORCH_OPTIM.keys())}")
    kwargs = {'lr': lr}
    if opt_type == 'sgd':
        kwargs['momentum'] = cfg_optimizer.get('momentum', 0.9)
        kwargs['nesterov'] = cfg_optimizer.get('nesterov', False)
    return cls(params, **kwargs)


# ─── Legacy MonoDLE custom AdamW ──────────────────────────────────────

class _LegacyAdamW(Optimizer):
    """Original MonoDLE custom AdamW (multiplicative weight decay).

    Kept strictly for backward compatibility / reproducing published results.
    For new experiments prefer ``type: adamw`` which uses ``torch.optim.AdamW``.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                amsgrad = group['amsgrad']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bc2) / bc1
                p.data.add_(
                    torch.mul(p.data, group['weight_decay']).addcdiv_(
                        exp_avg, denom, value=1
                    ),
                    alpha=-step_size,
                )
        return loss