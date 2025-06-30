import torch
import torch.nn as nn

class RMSNorm(torch.nn.Module):

    def __init__(self, d_model, epsilon=1e-5):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(data=torch.ones(d_model), requires_grad=True)
        self.epsilon = epsilon

    def load_state_dict(self, weights):
        self.weight = nn.Parameter(weights['weight'])

    def forward(self, x):
        rms = torch.sqrt((torch.mean(x.pow(2), axis=-1, keepdim=True)) + self.epsilon)
        return (x / rms) * self.weight
