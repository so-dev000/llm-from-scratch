import torch
from torch import nn

from layer.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, model_dim, encoder_num):
        super().__init__()
        self.encoders = nn.ModuleList(
            [EncoderLayer(model_dim) for _ in range(encoder_num)]
        )

    def forward(self, inputs):
        for encoder_layer in self.encoders:
            inputs = encoder_layer(inputs)
        return inputs


if __name__ == "__main__":
    batch_size = 3
    seq_len = 6
    model_dim = 512

    input = torch.randn(batch_size, seq_len, model_dim)
    encoder = Encoder(model_dim=model_dim, encoder_num=8)

    print(f"Input shape: {input.shape}")  # (3, 6, 512)
    output = encoder.forward(input)
    print(f"Output shape: {output.shape}")  # (3, 6, 512)
