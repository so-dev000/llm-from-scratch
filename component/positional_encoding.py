import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # positional encoding matrix
        pe = torch.zeros(max_len, model_dim)
        # position index (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # calc denominator (分母)
        # 10000^(2i/model_dim)
        #   = exp(log(10000^(2i/d_model)))
        #   =  exp((2i/d_model) * log(10000))
        denominator = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
        )
        # even index
        pe[:, 0::2] = torch.sin(position * denominator)
        # odd index
        pe[:, 1::2] = torch.cos(position * denominator)
        # add batch dim (max_len, model_dim) → (1, max_len, model_dim)
        pe = pe.unsqueeze(0)
        # register as buffer
        # cf: https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Buffer.html
        self.register_buffer("pe", pe)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, model_dim)
        seq_len = inputs.size(1)
        # slice positional encoding and add
        inputs = inputs + self.pe[:, :seq_len, :]
        return self.dropout(inputs)
