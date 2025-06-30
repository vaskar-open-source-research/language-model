import torch
import torch.nn as nn
from cs336_basics.language_model.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.language_model.rmsnorm import RMSNorm


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads:int , attn_pdrop: float | None = None, d_key: int | None = None, d_value: int | None = None):
        super(MultiHeadSelfAttention, self).__init__()
        if d_key is None:
            self.d_key = d_model // num_heads
        if d_value is None:
            self.d_value = d_model // num_heads
        self.d_model = d_model
        self.n_heads = num_heads
        self.q_proj = nn.Parameter(torch.randn((num_heads, self.d_key, d_model)))
        self.k_proj = nn.Parameter(torch.randn((num_heads, self.d_key, d_model)))
        self.v_proj = nn.Parameter(torch.randn((num_heads, self.d_value, d_model)))
        self.output_proj = nn.Parameter(torch.randn((d_model, num_heads * self.d_value)))
        self.attn_pdrop = attn_pdrop

    def load_state_dict(self, weights):
        self.q_proj = nn.Parameter(torch.stack([weights[f'q_heads.{n}.weight'] for n in range(self.n_heads)]))
        self.v_proj = nn.Parameter(torch.stack([weights[f'v_heads.{n}.weight'] for n in range(self.n_heads)]))
        self.k_proj = nn.Parameter(torch.stack([weights[f'k_heads.{n}.weight'] for n in range(self.n_heads)]))
        self.output_proj = nn.Parameter(weights['output_proj.weight'])

    def forward(self, x, cache=None):

        
        q = torch.matmul(x.unsqueeze(-3), self.q_proj.transpose(-2, -1))
        k = torch.matmul(x.unsqueeze(-3), self.k_proj.transpose(-2, -1))
        v = torch.matmul(x.unsqueeze(-3), self.v_proj.transpose(-2, -1))
        
        if cache is not None:
            k, v = cache.extend(k, v)
        
        mask = torch.triu(torch.ones((q.shape[-2], q.shape[-2]), device=q.device), diagonal=1).bool()
        attention_output = scaled_dot_product_attention(q, k, v, mask, self.attn_pdrop)
        
        shape = list(x.shape)
        shape[-1] = -1
        permute = [i for i in range(len(attention_output.shape))]
        permute[-2], permute[-3] = permute[-3], permute[-2]

        attention_output = attention_output.permute(permute).reshape(shape)
        
        return torch.matmul(attention_output, self.output_proj.transpose(-2, -1))
