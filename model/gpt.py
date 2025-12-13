import math

from torch import nn

from block.gpt_block import GPTBlock
from component.positional_embedding import PositionalEmbedding
from component.token_embedding import TokenEmbedding


# Default args: GPT-2 Small
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        model_dim=768,
        layer_num=12,
        head=12,
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
        self.layer_num = layer_num
        self.final_norm = nn.LayerNorm(model_dim)
        self.proj = nn.Linear(model_dim, vocab_size)
        # Weight Tying
        self.proj.weight = self.token_embedding.embedding.weight
        self._init_weight()

    # Weight modified initialization with residual scaling
    def _init_weight(self):
        scale = 1.0 / math.sqrt(2.0 * self.layer_num)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Scale residual projection layers
        for layer in self.gpt_block.gpt_layers:
            # Attention projection
            nn.init.normal_(
                layer.masked_attention.out_proj.weight,
                mean=0.0,
                std=0.02 * scale,
            )
            # FFN projection
            nn.init.normal_(
                layer.feed_forward.linear_2.weight,
                mean=0.0,
                std=0.02 * scale,
            )

    def forward(self, tokens, mask=None):
        # (batch_size, source_len, model_dim)
        token_embed = self.token_embedding(tokens)
        token_embed = self.positional_embedding(token_embed)
        token_embed = self.embedding_dropout(token_embed)

        gpt_out = self.gpt_block(token_embed, mask)
        gpt_out = self.final_norm(gpt_out)
        output = self.proj(gpt_out)
        return output
