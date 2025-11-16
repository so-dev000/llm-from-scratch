from torch import nn

from component.feed_forward import FeedForward
from component.multihead_attention import MultiheadAttention


class EncoderLayer(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.attention = MultiheadAttention(model_dim)
        self.normalizer_1 = nn.LayerNorm(model_dim)
        self.feed_forward = FeedForward(model_dim)
        self.normalizer_2 = nn.LayerNorm(model_dim)

    def forward(self, inputs, mask=None):
        attention_out = self.attention(inputs, mask=mask)
        normalized_1 = self.normalizer_1(inputs + attention_out)
        feed_forward_out = self.feed_forward(normalized_1)
        normalized_2 = self.normalizer_2(normalized_1 + feed_forward_out)
        return normalized_2
