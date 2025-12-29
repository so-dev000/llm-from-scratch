import torch
import torch.nn.functional as F
from torch import nn

from component.rotary_embedding import apply_rotary_emb


class Attention(nn.Module):
    def __init__(
        self, model_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        # num_kv_heads == num_heads: MHA
        # num_kv_heads == 1: MQA
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.wq = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)
        self.dropout_p = dropout

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x)  # (batch, seq, num_heads * head_dim)
        k = self.wk(x)  # (batch, seq, num_kv_heads * head_dim)
        v = self.wv(x)  # (batch, seq, num_kv_heads * head_dim)

        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # Expand K and V to match Q
        # (batch, seq, num_kv_heads, head_dim) → (batch, seq, num_heads, head_dim)
        k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        # (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Convert mask format for scaled_dot_product_attention
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                attn_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
            elif mask.dim() == 3:
                attn_mask = mask.unsqueeze(1)  # (batch, 1, seq, seq)
            else:
                attn_mask = mask
            # Ensure mask dtype matches query dtype
            attn_mask = attn_mask.to(dtype=q.dtype)

        # Use Flash Attention
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        output = output.transpose(1, 2)
        output = output.contiguous().view(batch_size, seq_len, -1)

        output = self.wo(output)

        return output
