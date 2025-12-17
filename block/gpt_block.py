from torch import nn

from layer.gpt_layer import GPTLayer


class GPTBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        layer_num: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        activation_func: str,
    ):
        super().__init__()
        self.gpt_layers = nn.ModuleList(
            [
                GPTLayer(
                    model_dim, num_heads, feedforward_dim, dropout, activation_func
                )
                for _ in range(layer_num)
            ]
        )

    def forward(self, inputs, mask=None):
        for gpt_layer in self.gpt_layers:
            inputs = gpt_layer(inputs, mask)
        return inputs
