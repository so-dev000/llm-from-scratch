from torch import nn

from layer.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, model_dim, encoder_num):
        super().__init__()
        self.encoders = nn.ModuleList(
            [EncoderLayer(model_dim) for _ in range(encoder_num)]
        )

    def forward(self, inputs, mask=None):
        for encoder_layer in self.encoders:
            inputs = encoder_layer(inputs, mask)
        return inputs
