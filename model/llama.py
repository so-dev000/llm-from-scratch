import torch
from torch import nn

from block.llama_block import LlamaBlock
from component.rms_norm import RMSNorm
from component.rotary_embedding import RotaryEmbedding
from component.token_embedding import TokenEmbedding


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(
            config.vocab_size, config.model_dim, config.padding_idx, scaling=False
        )
        self.rotary_embedding = RotaryEmbedding(
            config.model_dim // config.num_heads, config.max_seq_len, config.rope_theta
        )
        self.llama_block = LlamaBlock(
            config.model_dim,
            config.num_layers,
            config.num_heads,
            config.num_kv_heads,
            config.feedforward_dim,
            config.dropout,
            config.norm_eps,
            config.use_gradient_checkpointing,
        )
        self.norm = RMSNorm(config.model_dim, eps=config.norm_eps)
        self.output = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        # Weight tying
        self.output.weight = self.token_embedding.embedding.weight

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        # Token embedding
        x = self.token_embedding(tokens)

        # Get rotary embeddings
        seq_len = tokens.shape[1]
        freqs_cis = self.rotary_embedding.freqs_cis[:seq_len]

        x = self.llama_block(x, freqs_cis, mask)

        # Final norm
        x = self.norm(x)

        logits = self.output(x)

        return logits

    def prepare_context(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        return None

    def generate_next_token(self, tokens: torch.Tensor, context: torch.Tensor = None):
        # Truncate to max_seq_len if necessary
        if tokens.shape[1] > self.config.max_seq_len:
            tokens = tokens[:, -self.config.max_seq_len :]
        logits = self.forward(tokens, mask=None)
        return logits[:, -1, :]
