from torch import nn

from layer.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        encoder_num: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        activation_func: str,
    ):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                EncoderLayer(
                    model_dim, num_heads, feedforward_dim, dropout, activation_func
                )
                for _ in range(encoder_num)
            ]
        )

    def forward(self, inputs, mask=None):
        for encoder_layer in self.encoders:
            inputs = encoder_layer(inputs, mask)
        return inputs
