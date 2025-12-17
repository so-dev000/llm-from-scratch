from torch import nn

from layer.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        decoder_num: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        activation_func: str,
    ):
        super().__init__()
        self.decoders = nn.ModuleList(
            [
                DecoderLayer(
                    model_dim, num_heads, feedforward_dim, dropout, activation_func
                )
                for _ in range(decoder_num)
            ]
        )

    def forward(self, inputs, encoder_out, tgt_mask=None, src_mask=None):
        for decoder_layer in self.decoders:
            inputs = decoder_layer(inputs, encoder_out, tgt_mask, src_mask)
        return inputs
