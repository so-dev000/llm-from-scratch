from collections import OrderedDict

import torch
from torch import nn

from layer.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, model_dim, decoder_num):
        super().__init__()
        self.decoders = nn.Sequential(
            OrderedDict(
                [
                    (f"decoder_layer_{idx}", DecoderLayer(model_dim))
                    for idx in range(decoder_num)
                ]
            )
        )

    def forward(self, inputs):
        return self.decoders(inputs)


if __name__ == "__main__":
    batch_size = 3
    seq_len = 6
    model_dim = 512

    encoder_out = torch.randn(batch_size, seq_len, model_dim)
    input = torch.randn(batch_size, seq_len, model_dim)
    decoder = Decoder(model_dim=model_dim, decoder_num=8)

    print(f"Input shape: {input.shape}")  # (3, 6, 512)
    output = decoder.forward(input, encoder_out)
    print(f"Output shape: {output.shape}")  # (3, 6, 512)
