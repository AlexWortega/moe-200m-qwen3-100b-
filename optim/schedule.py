"""WSD learning-rate schedule + EMA wrapper.

WSD = Warmup → Steady → Decay.
  - warmup over [0, w_steps]: linear from 0 to peak_lr
  - steady over [w_steps, decay_start]: constant peak_lr
  - decay over [decay_start, total_steps]: 1/sqrt(progress), reaching min_lr
"""
import math
from typing import Optional

import torch


class WSDSchedule:
    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int,
                 decay_frac: float = 0.2, min_lr: float = 1e-5,
                 decay_shape: str = "sqrt", decay_steps: Optional[int] = None):
        self.peak_lr = peak_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = total_steps
        if decay_steps is not None:
            self.decay_start = max(self.warmup_steps + 1, total_steps - decay_steps)
        else:
            self.decay_start = max(self.warmup_steps + 1, int(total_steps * (1 - decay_frac)))
        self.min_lr = min_lr
        assert decay_shape in ("sqrt", "linear"), decay_shape
        self.decay_shape = decay_shape

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * (step / self.warmup_steps)
        if step < self.decay_start:
            return self.peak_lr
        progress = (step - self.decay_start) / max(1, self.total_steps - self.decay_start)
        progress = min(1.0, progress)
        if self.decay_shape == "linear":
            lr = self.peak_lr * (1.0 - progress) + self.min_lr * progress
        else:
            lr = self.peak_lr * (1.0 - math.sqrt(progress)) + self.min_lr * math.sqrt(progress)
        return max(lr, self.min_lr)

    def is_in_decay(self, step: int) -> bool:
        return step >= self.decay_start

    def state_dict(self):
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "decay_start": self.decay_start,
                "min_lr": self.min_lr, "decay_shape": self.decay_shape}

    def load_state_dict(self, sd):
        self.peak_lr = sd["peak_lr"]
        self.warmup_steps = sd["warmup_steps"]
        self.total_steps = sd["total_steps"]
        self.decay_start = sd["decay_start"]
        self.min_lr = sd["min_lr"]
        self.decay_shape = sd.get("decay_shape", "sqrt")


class EMA:
    """Exponential-moving-average shadow weights. Only updated when activated
    (plan: late-only). Lives on the same device as the model."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.activated = False
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def activate(self, model: torch.nn.Module):
        self.activated = True
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].copy_(p.detach())

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        if not self.activated:
            return
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow[n]
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_into(self, model: torch.nn.Module):
        """Copy shadow into model, returning the original weights so they can be
        restored later."""
        backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                backup[n] = p.detach().clone()
                p.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.copy_(backup[n])

    def state_dict(self):
        return {"decay": self.decay, "activated": self.activated,
                "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, sd, device="cuda"):
        self.decay = sd["decay"]
        self.activated = sd["activated"]
        for k, v in sd["shadow"].items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))
