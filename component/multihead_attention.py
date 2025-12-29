from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        # dimension of Q(query), K(key), V(value) weights
        self.head_dim = model_dim // num_heads
        # number of head
        self.num_heads = num_heads
        # projections for Q, K, V: (model_dim, self.head_dim * num_heads = model_dim)
        self.wq = nn.Linear(model_dim, self.head_dim * num_heads, bias=True)
        self.wk = nn.Linear(model_dim, self.head_dim * num_heads, bias=True)
        self.wv = nn.Linear(model_dim, self.head_dim * num_heads, bias=True)
        # output projection: (self.head_dim * num_heads = model_dim, model_dim)
        self.wo = nn.Linear(self.head_dim * num_heads, model_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, encoder_out=None, mask=None):
        # input: (batch_size, seq_len, model_dim)
        batch_size, seq_len, _ = x.shape

        # Projection: (batch_size, seq_len, model_dim)
        # → (batch_size, seq_len, model_dim)
        query = self.wq(x)
        key = self.wk(encoder_out if encoder_out is not None else x)
        value = self.wv(encoder_out if encoder_out is not None else x)

        # Reshape and transpose for multi-head
        # (batch_size, seq_len, model_dim)
        # → (batch_size, seq_len, head, head_dim)
        # → (batch_size, head, seq_len, head_dim)
        # → (batch_size, seq_len, head, head_dim)
        # → (batch_size, head, seq_len, head_dim)
        query = query.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Q x K^T: (batch_size, head, seq_len, seq_len)
        # key.transpose(-2, -1): (batch_size, head, head_dim, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1))

        # divide by square root of key dimension
        scores /= sqrt(self.head_dim)

        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                if encoder_out is not None:  # Cross-attention, assume padding mask
                    mask = mask.unsqueeze(1).unsqueeze(2)
                else:  # Self-attention, assume causal mask
                    mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:  # attention mask (batch, query_seq_len, key_seq_len)
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, -1e9)

        # softmax: (batch_size, head, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)

        # dropout
        attention_weights = self.dropout(attention_weights)

        # z: (batch_size, head, seq_len, head_dim)
        z = torch.matmul(attention_weights, value)

        # (batch_size, head, seq_len, head_dim)
        # → (batch_size, seq_len, head, head_dim)
        # cf: https://qiita.com/kenta1984/items/d68b72214ce92beebbe2
        z = z.transpose(1, 2).contiguous()

        # (batch_size, seq_len, head, head_dim)
        # → (batch_size, seq_len, head * head_dim)
        z = z.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # output projection: (batch_size, seq_len, model_dim)
        output = self.wo(z)

        # dropout
        output = self.dropout(output)
        return output
