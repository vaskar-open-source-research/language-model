import torch.nn as nn
import torch.nn.functional as F
from src.language_model.multihead_self_attention import MultiHeadSelfAttention
from src.language_model.ffn import FFN
from src.language_model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_pdrop: float | None = None, residual_pdrop: float | None = None):
        super(TransformerBlock, self).__init__()
        self.num_heads, self.d_model, self.d_ff = num_heads, d_model, d_ff
        self.attn = MultiHeadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = FFN(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.residual_pdrop = residual_pdrop

    def load_state_dict(self, weights):
        attn_q_proj_weights = weights['attn.q_proj.weight']
        attn_k_proj_weights = weights['attn.k_proj.weight']
        attn_v_proj_weights = weights['attn.v_proj.weight']
        attn_out_proj_weights = weights['attn.output_proj.weight']

        mha_weights = {}
        for head_type, proj_weights in [
            ('q', attn_q_proj_weights),
            ('k', attn_k_proj_weights),
            ('v', attn_v_proj_weights)
        ]:
            for i in range(self.num_heads):
                start_idx = i * (self.d_model // self.num_heads)
                end_idx = (i + 1) * (self.d_model // self.num_heads)
                mha_weights[f'{head_type}_heads.{i}.weight'] = proj_weights.reshape(-1, self.d_model)[start_idx:end_idx, :]

        mha_weights['output_proj.weight'] = attn_out_proj_weights

        self.attn.load_state_dict(mha_weights)
        self.ffn.load_state_dict(
            {
                'w1.weight': weights['ffn.w1.weight'],
                'w2.weight': weights['ffn.w2.weight'],
            }
        )
        self.ln1.load_state_dict({
            'weight': weights['ln1.weight']
        })
        self.ln2.load_state_dict({
            'weight': weights['ln2.weight']
        })

    def forward(self, x, cache=None):
        x = x + F.dropout(self.attn(self.ln1(x), cache), self.residual_pdrop)
        x = x + F.dropout(self.ffn(self.ln2(x)), self.residual_pdrop)
        return x
        