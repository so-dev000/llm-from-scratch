import torch
from torch import nn

from layer.llama_layer import LlamaLayer


class LlamaBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LlamaLayer(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    feedforward_dim,
                    dropout,
                    norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, mask: torch.Tensor = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        return x
