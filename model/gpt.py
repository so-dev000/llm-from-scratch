import math

from torch import nn

from block.gpt_block import GPTBlock
from component.positional_embedding import PositionalEmbedding
from component.token_embedding import TokenEmbedding


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_num = config.num_layers

        self.token_embedding = TokenEmbedding(
            config.vocab_size, config.model_dim, config.padding_idx, scaling=False
        )
        self.positional_embedding = PositionalEmbedding(
            config.max_seq_len, config.model_dim
        )
        self.embedding_dropout = nn.Dropout(p=config.dropout)
        self.gpt_block = GPTBlock(
            config.model_dim,
            config.num_layers,
            config.num_heads,
            config.feedforward_dim,
            config.dropout,
            config.activation,
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.proj = nn.Linear(config.model_dim, config.vocab_size)
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

    def prepare_context(self, tokens, mask=None):
        raise NotImplementedError()

    def generate_next_token(self, tokens, context=None):
        raise NotImplementedError()
