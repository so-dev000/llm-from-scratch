import torch
from torch import nn

from component.feed_forward import FeedForward
from component.multihead_attention import MultiheadAttention


class DecoderLayer(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.masked_attention = MultiheadAttention(model_dim, mask=True)
        self.normalizer_1 = nn.LayerNorm(model_dim)
        self.attention = MultiheadAttention(model_dim)
        self.normalizer_2 = nn.LayerNorm(model_dim)
        self.feed_forward = FeedForward(model_dim)
        self.normalizer_3 = nn.LayerNorm(model_dim)

    def forward(self, inputs, encoder_out):
        masked_attention_out = self.masked_attention(inputs)
        normalized_1 = self.normalizer_1(inputs + masked_attention_out)
        attention_out = self.attention(normalized_1, encoder_out)
        normalized_2 = self.normalizer_2(normalized_1 + attention_out)
        feed_forward_out = self.feed_forward(normalized_2)
        normalized_3 = self.normalizer_3(normalized_2 + feed_forward_out)
        return normalized_3


if __name__ == "__main__":
    batch_size = 3
    seq_len = 6
    model_dim = 512

    encoder_out = torch.randn(batch_size, seq_len, model_dim)
    input = torch.randn(batch_size, seq_len, model_dim)
    encoder = DecoderLayer(model_dim=model_dim)

    print(f"Input shape: {input.shape}")  # (3, 6, 512)
    output = encoder.forward(input, encoder_out)
    print(f"Output shape: {output.shape}")  # (3, 6, 512)
