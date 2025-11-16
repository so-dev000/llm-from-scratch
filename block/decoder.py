from torch import nn

from layer.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, model_dim, decoder_num):
        super().__init__()
        self.decoders = nn.ModuleList(
            [DecoderLayer(model_dim) for _ in range(decoder_num)]
        )

    def forward(self, inputs, encoder_out, tgt_mask=None, src_mask=None):
        for decoder_layer in self.decoders:
            inputs = decoder_layer(inputs, encoder_out, tgt_mask, src_mask)
        return inputs
