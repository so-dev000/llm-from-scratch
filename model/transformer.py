import torch
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
        encoder_out = self.encode(source_tokens, encoder_src_mask)
        decoder_out = self.decode(
            target_tokens, encoder_out, tgt_mask, decoder_src_mask
        )
        return self.decoder_proj(decoder_out)

    def encode(self, source_tokens, src_mask):
        src_embed = self.src_embedding(source_tokens)
        src_embed = self.positional_encoding(src_embed)
        return self.encoder(src_embed, src_mask)

    def decode(self, target_tokens, encoder_out, tgt_mask, src_mask):
        tgt_embed = self.tgt_embedding(target_tokens)
        tgt_embed = self.positional_encoding(tgt_embed)
        return self.decoder(tgt_embed, encoder_out, tgt_mask, src_mask)

    def prepare_context(self, source_tokens, src_mask=None):
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1) & src_mask.unsqueeze(2)
        encoder_out = self.encode(source_tokens, src_mask)
        return {"encoder_out": encoder_out, "src_mask": src_mask}

    def generate_next_token(self, target_tokens, context):
        encoder_out = context["encoder_out"]
        src_mask = context["src_mask"]
        tgt_len = target_tokens.size(1)
        tgt_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=target_tokens.device)
        ).bool()
        decoder_out = self.decode(target_tokens, encoder_out, tgt_mask, src_mask)
        return self.decoder_proj(decoder_out[:, -1, :])
