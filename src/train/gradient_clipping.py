import torch

def gradient_clipping(parameters, max_l2_norm):
    l2_norm_sum = 0
    for param in parameters:
        if param.grad is not None:
            l2_norm_sum += torch.sum(torch.square(param.grad.data))
        else:
            continue
    
    l2_norm = torch.sqrt(l2_norm_sum)
    for param in parameters:
        if param.grad is not None:
            if l2_norm > max_l2_norm:
                param.grad.data *= (max_l2_norm / (l2_norm + 1e-6))

    return l2_norm.item()