import torch.nn as nn
from src.language_model.gelu import gelu

class FFN(nn.Module):

    def __init__(self, d_model, d_ff):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def load_state_dict(self, weights):
        self.w1.weight = nn.Parameter(weights['w1.weight'])
        self.w2.weight = nn.Parameter(weights['w2.weight'])

    def forward(self, x):
        return self.w2(gelu(self.w1(x)))
        