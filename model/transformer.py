from torch import nn

from block.decoder import Decoder
from block.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, model_dim, encoder_num, decoder_num):
        super().__init__()
        self.model_dim = model_dim
        self.encoder_num = encoder_num
        self.decoder_num = decoder_num
        self.encoder = Encoder(self.model_dim, self.encoder_num)
        self.decoder = Decoder(self.model_dim, self.decoder_num)

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_out = self.encoder(encoder_inputs)
        decoder_out = self.decoder(decoder_inputs, encoder_out)
        return decoder_out
