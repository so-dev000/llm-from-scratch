from torch import nn

from component.feed_forward import FeedForward
from component.multihead_attention import MultiheadAttention


class GPTLayer(nn.Module):
    def __init__(self, model_dim, head):
        super().__init__()
        self.masked_attention = MultiheadAttention(model_dim, head)
        self.normalizer_1 = nn.LayerNorm(model_dim)
        self.feed_forward = FeedForward(model_dim, nn.GELU())
        self.normalizer_2 = nn.LayerNorm(model_dim)

    def forward(self, inputs, mask=None):
        # Pre-LN: Normalize before attention
        normalized_1 = self.normalizer_1(inputs)
        masked_attention_out = self.masked_attention(normalized_1, mask=mask)
        x = inputs + masked_attention_out  # Residual

        # Pre-LN: Normalize before FFN
        normalized_2 = self.normalizer_2(x)
        feed_forward_out = self.feed_forward(normalized_2)
        output = x + feed_forward_out  # Redisual
        return output
