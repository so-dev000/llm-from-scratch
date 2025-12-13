import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, model_dim):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, model_dim)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, model_dim)
        batch_size, seq_len, model_dim = inputs.shape
        # position index (1, seq_len)
        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0)
        # position index (batch_size, seq_len)
        positions = positions.expand(batch_size, -1)
        position_embed = self.embedding(positions)
        return inputs + position_embed
