import time
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(self, model_dim, head=8):
        super().__init__()
        # dimension of Q(query), K(key), V(value) weights
        self.w_qkv_dim = model_dim // head
        # number of head
        self.head = head
        # QKV projection
        self.qkv_proj = nn.Linear(model_dim, self.w_qkv_dim * head * 3, bias=True)
        # output projection
        self.out_proj = nn.Linear(self.w_qkv_dim * head, model_dim, bias=True)

    def forward(self, input):
        # input: (batch_size, seq_len, model_dim)
        batch_size, seq_len, _ = input.shape

        # input · self.weight
        qkv = self.qkv_proj(input)

        # (batch_size, seq_len, w_qkv_dim * head * 3) → (batch_size, seq_len, 3, head, w_qkv_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.head, self.w_qkv_dim)

        # (batch_size, seq_len, 3, head, w_qkv_dim) → (3, batch_size, head, seq_len, w_qkv_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # split into (batch_size, head, seq_len, w_qkv_dim) x 3
        # query: Q_0, ··· , Q_(head-1)
        # key:   K_0, ··· , K_(head-1)
        # value: V_0, ··· , V_(head-1)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Q x K^T: (batch_size, head, seq_len, seq_len)
        # key.transpose(-2, -1): (batch_size, head, w_qkv_dim, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1))

        # divide by square root of key dimension
        scores /= sqrt(self.w_qkv_dim)

        # softmax: (batch_size, head, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)

        # z: (batch_size, head, seq_len, w_qkv_dim)
        z = torch.matmul(attention_weights, value)

        # (batch_size, head, seq_len, w_qkv_dim) → (batch_size, seq_len, head, w_qkv_dim)
        z = z.transpose(1, 2)

        # (batch_size, seq_len, head, w_qkv_dim) → (batch_size, seq_len, head * w_qkv_dim)
        z = z.reshape(batch_size, seq_len, self.head * self.w_qkv_dim)

        # output projection: (batch_size, seq_len, model_dim)
        output = self.out_proj(z)
        return output


if __name__ == "__main__":
    start = time.perf_counter()
    batch_size = 3
    seq_len = 6
    model_dim = 512

    input = torch.randn(batch_size, seq_len, model_dim)
    model = MultiheadAttention(model_dim=model_dim)

    print(f"Input shape: {input.shape}")  # (3, 6, 512)
    output = model.forward(input)
    print(f"Output shape: {output.shape}")  # (3, 6, 512)
    end = time.perf_counter()
    print(end - start)
