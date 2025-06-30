import torch
import math

def gelu(x):
    return (x / 2) * (1 + torch.erf(x / math.sqrt(2)))