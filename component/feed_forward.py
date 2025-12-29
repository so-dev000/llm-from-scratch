from typing import Literal

from torch import nn

ACTIVATION_FUNC = Literal["relu", "gelu"]


class FeedForward(nn.Module):
    def __init__(
        self,
        model_dim: int,
        feedforward_dim: int,
        dropout: float,
        activation_func: ACTIVATION_FUNC,
    ):
        super().__init__()
        self.w1 = nn.Linear(model_dim, feedforward_dim)
        if activation_func.lower() == "relu":
            self.activation_func = nn.ReLU()
        elif activation_func.lower() == "gelu":
            self.activation_func = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {activation_func}")
        self.w2 = nn.Linear(feedforward_dim, model_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h = self.w1(x)
        h = self.activation_func(h)
        output = self.w2(h)
        # dropout
        output = self.dropout(output)
        return output
