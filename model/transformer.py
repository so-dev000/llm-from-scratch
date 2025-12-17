from torch import nn

from block.decoder import Decoder
from block.encoder import Encoder
from component.positional_encoding import PositionalEncoding
from component.token_embedding import TokenEmbedding


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.src_embedding = TokenEmbedding(
            config.src_vocab_size, config.model_dim, config.padding_idx
        )
        self.tgt_embedding = TokenEmbedding(
            config.tgt_vocab_size, config.model_dim, config.padding_idx
        )
        self.positional_encoding = PositionalEncoding(
            config.model_dim, config.dropout, config.max_seq_len
        )
        self.encoder = Encoder(
            config.model_dim,
            config.encoder_layers,
            config.num_heads,
            config.feedforward_dim,
            config.dropout,
            config.activation,
        )
        self.decoder = Decoder(
            config.model_dim,
            config.decoder_layers,
            config.num_heads,
            config.feedforward_dim,
            config.dropout,
            config.activation,
        )
        self.decoder_proj = nn.Linear(config.model_dim, config.tgt_vocab_size)

    def forward(
        self,
        source_tokens,
        target_tokens,
        encoder_src_mask=None,
        decoder_src_mask=None,
        tgt_mask=None,
    ):
        # (batch_size, source_len, model_dim)
        source_embed = self.src_embedding(source_tokens)
        source_embed = self.positional_encoding(source_embed)
        # (batch_size, target, model_dim)
        target_embed = self.tgt_embedding(target_tokens)
        target_embed = self.positional_encoding(target_embed)

        encoder_out = self.encoder(source_embed, encoder_src_mask)
        decoder_out = self.decoder(
            target_embed, encoder_out, tgt_mask, decoder_src_mask
        )
        output = self.decoder_proj(decoder_out)
        return output
