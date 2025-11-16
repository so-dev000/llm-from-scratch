from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(self, model_dim, head=8, dropout=0.1):
        super().__init__()
        if model_dim % head != 0:
            raise ValueError("model_dim must be divisible by head")
        # dimension of Q(query), K(key), V(value) weights
        self.w_qkv_dim = model_dim // head
        # number of head
        self.head = head
        # projections for Q, K, V: (model_dim, self.w_qkv_dim * head = model_dim)
        self.q_proj = nn.Linear(model_dim, self.w_qkv_dim * head, bias=True)
        self.k_proj = nn.Linear(model_dim, self.w_qkv_dim * head, bias=True)
        self.v_proj = nn.Linear(model_dim, self.w_qkv_dim * head, bias=True)
        # output projection: (self.w_qkv_dim * head = model_dim, model_dim)
        self.out_proj = nn.Linear(self.w_qkv_dim * head, model_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, encoder_out=None, mask=None):
        # input: (batch_size, seq_len, model_dim)
        batch_size, seq_len, _ = inputs.shape

        # Projection: (batch_size, seq_len, model_dim)
        # → (batch_size, seq_len, model_dim)
        query = self.q_proj(inputs)
        key = self.k_proj(encoder_out if encoder_out is not None else inputs)
        value = self.v_proj(encoder_out if encoder_out is not None else inputs)

        # Reshape and transpose for multi-head
        # (batch_size, seq_len, model_dim)
        # → (batch_size, seq_len, head, w_qkv_dim)
        # → (batch_size, head, seq_len, w_qkv_dim)
        query = query.view(batch_size, seq_len, self.head, self.w_qkv_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.head, self.w_qkv_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.head, self.w_qkv_dim).transpose(1, 2)

        # Q x K^T: (batch_size, head, seq_len, seq_len)
        # key.transpose(-2, -1): (batch_size, head, w_qkv_dim, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1))

        # divide by square root of key dimension
        scores /= sqrt(self.w_qkv_dim)

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

        # z: (batch_size, head, seq_len, w_qkv_dim)
        z = torch.matmul(attention_weights, value)

        # (batch_size, head, seq_len, w_qkv_dim)
        # → (batch_size, seq_len, head, w_qkv_dim)
        # cf: https://qiita.com/kenta1984/items/d68b72214ce92beebbe2
        z = z.transpose(1, 2).contiguous()

        # (batch_size, seq_len, head, w_qkv_dim)
        # → (batch_size, seq_len, head * w_qkv_dim)
        z = z.view(batch_size, seq_len, self.head * self.w_qkv_dim)

        # output projection: (batch_size, seq_len, model_dim)
        output = self.out_proj(z)

        # dropout
        output = self.dropout(output)
        return output
