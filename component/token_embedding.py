from math import sqrt

from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=padding_idx)
        self.model_dim = model_dim

    def forward(self, inputs):
        return self.embedding(inputs) * sqrt(self.model_dim)
