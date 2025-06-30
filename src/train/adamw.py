from collections.abc import Callable 
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, weight_decay, lr=1e-5, betas=(0.9, 0.999), eps=1e-8):
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "decay": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults)

    def update_lr(self, lr):
        for group in self.param_groups:
            group["lr"] = lr
    
    def get_t(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                return state.get("t", 1)
        return 1

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, beta1, beta2, decay, eps = group["lr"], group["beta1"], group["beta2"], group["decay"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)
                g = p.grad.data
                m = (beta1 * m) + (1 - beta1) * g
                v = (beta2 * v) + (1 - beta2) * torch.square(g)
                curr_lr = lr * ((math.sqrt(1 - math.pow(beta2, t))) / (1 - math.pow(beta1, t)))
                p.data = p.data - (curr_lr * (m / (torch.sqrt(v) + eps)))
                p.data = p.data - (lr * decay * p.data)
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return {"lr": lr}
