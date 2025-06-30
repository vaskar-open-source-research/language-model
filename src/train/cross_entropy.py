import torch

def cross_entropy(logits, targets):
    m = torch.max(logits, dim=-1).values.unsqueeze(-1)
    return torch.mean(torch.gather(-(logits - m) + torch.log(torch.sum(torch.exp(logits - m), dim=-1, keepdim=True)), dim=-1, index=targets.unsqueeze(-1)))