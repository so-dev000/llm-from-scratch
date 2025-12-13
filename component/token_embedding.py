from math import sqrt

from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, padding_idx=None, scaling=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=padding_idx)
        self.model_dim = model_dim
        self.scaling = scaling

    def forward(self, inputs):
        if self.scaling:
            return self.embedding(inputs) * sqrt(self.model_dim)
        return self.embedding(inputs)
