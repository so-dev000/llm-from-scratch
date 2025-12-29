import torch
from torch import nn

from component.attention import Attention
from component.feed_forward_swiglu import FeedForwardSwiGLU
from component.rms_norm import RMSNorm


class LlamaLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attention = Attention(model_dim, num_heads, num_kv_heads, dropout)
        self.feed_forward = FeedForwardSwiGLU(model_dim, feedforward_dim, dropout)
        self.attention_norm = RMSNorm(model_dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(model_dim, eps=norm_eps)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, mask: torch.Tensor = None
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        raise out
