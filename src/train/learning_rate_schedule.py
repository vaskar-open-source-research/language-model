import torch
import math

def learning_rate_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):

    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + ((1/2) * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) *  math.pi)) * (max_learning_rate - min_learning_rate))


class WarmupCosine:
    def __init__(self, optimizer, max_learning_rate, min_learning_rate, warmup_steps, cosine_cycle_iters):
        self.optimizer = optimizer
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_steps = warmup_steps
        self.cosine_cycle_iters = cosine_cycle_iters
        self.step()
        
    def get_lr(self):
        return learning_rate_schedule(self.optimizer.get_t(), self.max_learning_rate, self.min_learning_rate, self.warmup_steps, self.cosine_cycle_iters)
    
    def step(self):
        self.optimizer.update_lr(self.get_lr())
