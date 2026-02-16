"""
Exponential Moving Average (EMA) for model parameters.

Maintains a shadow copy of model parameters updated with an exponential
moving average.  The EMA model typically generalises better than the raw
training model and is used for evaluation / final deployment.

Usage::

    ema = ModelEMA(model, decay=0.9999)
    for batch in loader:
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        ema.update(model)
    # evaluate with EMA weights
    ema.apply_shadow(model)
"""

import math
import copy
from collections import OrderedDict

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay factor (default: 0.9999).
        warmup_steps: Number of steps over which the decay ramps up from 0
            to ``decay`` (default: 2000).  Set to 0 to disable warmup.
        device: Device for the shadow parameters (default: same as model).
    """

    def __init__(self, model, decay=0.9999, warmup_steps=2000, device=None):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0
        # Deep copy the model parameters as shadow
        self.shadow = OrderedDict()
        self.device = device
        self._init_shadow(model)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_shadow(self, model):
        """Create a deep copy of model parameters as shadow."""
        m = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
        for name, param in m.state_dict().items():
            if self.device is not None:
                self.shadow[name] = param.clone().detach().to(self.device)
            else:
                self.shadow[name] = param.clone().detach()

    def _get_decay(self):
        """Compute effective decay with optional warmup ramp."""
        self.updates += 1
        if self.warmup_steps > 0 and self.updates <= self.warmup_steps:
            # Ramp from 0 → decay over warmup_steps
            return self.decay * (1 - math.exp(-self.updates / self.warmup_steps))
        return self.decay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, model):
        """Update shadow parameters with EMA of current model parameters."""
        decay = self._get_decay()
        m = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
        model_state = m.state_dict()
        for name, shadow_param in self.shadow.items():
            if name in model_state:
                model_param = model_state[name]
                if model_param.dtype.is_floating_point:
                    shadow_param.lerp_(model_param.to(shadow_param.device), 1.0 - decay)
                else:
                    shadow_param.copy_(model_param)

    def apply_shadow(self, model):
        """Copy EMA (shadow) parameters into the model."""
        m = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
        m.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        """Restore original model parameters (undo ``apply_shadow``)."""
        raise RuntimeError(
            "restore() is not supported — keep a backup of model state_dict "
            "before calling apply_shadow() if you need to revert."
        )

    def state_dict(self):
        """Return serialisable state."""
        return {
            'decay': self.decay,
            'warmup_steps': self.warmup_steps,
            'updates': self.updates,
            'shadow': self.shadow,
        }

    def load_state_dict(self, state):
        """Restore from a previously saved ``state_dict()``."""
        self.decay = state['decay']
        self.warmup_steps = state['warmup_steps']
        self.updates = state['updates']
        self.shadow = state['shadow']
