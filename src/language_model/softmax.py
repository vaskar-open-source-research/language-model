import torch

def softmax(x, dim=-1):
    m = torch.max(x, dim=dim).values.unsqueeze(dim)
    x_exp = torch.exp(x - m)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
