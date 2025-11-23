from torch import nn

from block.decoder import Decoder
from block.encoder import Encoder
from component.positional_encoding import PositionalEncoding
from component.token_embedding import TokenEmbedding


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        model_dim,
        encoder_num,
        decoder_num,
        padding_idx=None,
    ):
        super().__init__()
        self.src_embedding = TokenEmbedding(src_vocab_size, model_dim, padding_idx)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, model_dim, padding_idx)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.encoder = Encoder(model_dim, encoder_num)
        self.decoder = Decoder(model_dim, decoder_num)
        self.decoder_proj = nn.Linear(model_dim, tgt_vocab_size)

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
