from torch import nn

from component.feed_forward import FeedForward
from component.multihead_attention import MultiheadAttention


class DecoderLayer(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.masked_attention = MultiheadAttention(model_dim)
        self.normalizer_1 = nn.LayerNorm(model_dim)
        self.attention = MultiheadAttention(model_dim)
        self.normalizer_2 = nn.LayerNorm(model_dim)
        self.feed_forward = FeedForward(model_dim)
        self.normalizer_3 = nn.LayerNorm(model_dim)

    def forward(self, inputs, encoder_out, tgt_mask=None, src_mask=None):
        masked_attention_out = self.masked_attention(inputs, mask=tgt_mask)
        normalized_1 = self.normalizer_1(inputs + masked_attention_out)
        attention_out = self.attention(normalized_1, encoder_out, mask=src_mask)
        normalized_2 = self.normalizer_2(normalized_1 + attention_out)
        feed_forward_out = self.feed_forward(normalized_2)
        normalized_3 = self.normalizer_3(normalized_2 + feed_forward_out)
        return normalized_3
