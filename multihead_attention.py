from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

# TODO: for文の効率が悪いのでテンソル演算に置き換える


class MultiheadAttention(nn.Module):
    def __init__(self, model_dim, head=8):
        super().__init__()
        # dimension of Q(query), K(key), V(value) weights
        self.w_qkv_dim = model_dim // head
        # number of head
        self.head = head
        # weight (model_dim, w_qkv_dim * head * 3)
        # → W_0^Q, ··· , W_(head-1)^Q, W_0^K, ··· , W_(head-1)^K, W_0^V, ··· , W_(head-1)^V
        self.weight = nn.Parameter(torch.randn(model_dim, self.w_qkv_dim * head * 3))
        # bias (w_qkv_dim * head * 3)
        self.bias = nn.Parameter(torch.randn(self.w_qkv_dim * head * 3))
        # output projection weight
        self.w_o = nn.Parameter(torch.randn(self.w_qkv_dim * head, model_dim))

    def forward(self, input):
        # input: (batch_size, seq_len, model_dim)

        # input · self.weight
        qkv = torch.matmul(input, self.weight) + self.bias
        # split into (batch_size, seq_len, w_qkv_dim * head) x 3
        # query: Q_0, ··· , Q_(head-1)
        # key:   K_0, ··· , K_(head-1)
        # value: V_0, ··· , V_(head-1)
        query, key, value = torch.split(qkv, self.w_qkv_dim * self.head, dim=-1)

        # list of z calculated in each head
        z_list = []
        for idx in range(self.head):
            # get Q, K, V for each head: (batch_size, seq_len, w_qkv_dim)
            q_head = query[
                :, :, idx * self.w_qkv_dim : (idx + 1) * self.w_qkv_dim
            ]  # Q_idx
            k_head = key[
                :, :, idx * self.w_qkv_dim : (idx + 1) * self.w_qkv_dim
            ]  # K_idx
            v_head = value[
                :, :, idx * self.w_qkv_dim : (idx + 1) * self.w_qkv_dim
            ]  # V_idx

            # Q x K^T: (batch_size, seq_len, seq_len)
            # k_head.transpose(-2, -1): (batch_size, w_qkv_dim, seq_len)
            q_kt = torch.matmul(q_head, k_head.transpose(-2, -1))

            # divide by square root of key dimension
            q_kt /= sqrt(self.w_qkv_dim)
            # softmax
            softmax_q_kt = F.softmax(q_kt, dim=-1)
            # z: (batch_size, seq_len, w_qkv_dim)
            z = torch.matmul(softmax_q_kt, v_head)
            z_list.append(z)

        # concatenate all heads: (batch_size, seq_len, w_qkv_dim * head)
        z_concat = torch.cat(z_list, dim=-1)
        # output projection: (batch_size, seq_len, model_dim)
        output = torch.matmul(z_concat, self.w_o)
        return output


if __name__ == "__main__":
    batch_size = 3
    seq_len = 6
    model_dim = 512

    input = torch.randn(batch_size, seq_len, model_dim)
    model = MultiheadAttention(model_dim=model_dim)

    print(f"Input shape: {input.shape}")  # (3, 6, 512)
    output = model.forward(input)
    print(f"Output shape: {output.shape}")  # (3, 6, 512)
