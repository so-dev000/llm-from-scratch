from torch import nn

from component.feed_forward import FeedForward
from component.multihead_attention import MultiheadAttention


class GPTLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        activation_func: str,
    ):
        super().__init__()
        self.attention = MultiheadAttention(model_dim, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feed_forward = FeedForward(
            model_dim, feedforward_dim, dropout, activation_func
        )
        self.ffn_norm = nn.LayerNorm(model_dim)

    def forward(self, x, mask=None):
        # Pre-LN: Normalize before attention
        h = self.attention_norm(x)
        h = self.attention(h, mask=mask)
        h = x + h  # Residual

        # Pre-LN: Normalize before FFN
        output = self.ffn_norm(h)
        output = self.feed_forward(output)
        output = h + output  # Residual
        return output
