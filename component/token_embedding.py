from math import sqrt

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size=(vocab_size, model_dim)))
        self.model_dim = model_dim

    def forward(self, inputs):
        return self.weight[inputs] * sqrt(self.model_dim)
