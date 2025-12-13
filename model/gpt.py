from torch import nn

from block.gpt_block import GPTBlock
from component.positional_embedding import PositionalEmbedding
from component.token_embedding import TokenEmbedding


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        model_dim,
        layer_num,
        head,
        max_seq_len=1024,
        padding_idx=None,
        dropout=0.1,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(
            vocab_size, model_dim, padding_idx, scaling=False
        )
        self.positional_embedding = PositionalEmbedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(p=dropout)
        self.gpt_block = GPTBlock(model_dim, layer_num, head)
        self.final_norm = nn.LayerNorm(model_dim)
        self.proj = nn.Linear(model_dim, vocab_size)

    def forward(self, tokens, mask=None):
        # (batch_size, source_len, model_dim)
        token_embed = self.token_embedding(tokens)
        token_embed = self.positional_embedding(token_embed)
        token_embed = self.embedding_dropout(token_embed)

        gpt_out = self.gpt_block(token_embed, mask)
        gpt_out = self.final_norm(gpt_out)
        output = self.proj(gpt_out)
        return output
