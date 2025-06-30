import torch
import torch.nn.functional as F
import math
from cs336_basics.language_model.softmax import softmax

def scaled_dot_product_attention(q, k, v, mask=None, pdrop=None):

    d_model = k.shape[-1]
    attention_scores = torch.matmul(q, torch.transpose(k, -2, -1)) / math.sqrt(float(d_model))
    if mask is not None:
        attention_scores += torch.where(mask, float('-inf'), 0)
    if pdrop is not None:
        attention_scores = F.dropout(softmax(attention_scores), pdrop)
    return torch.matmul(attention_scores, v)
