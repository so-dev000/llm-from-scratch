from typing import Literal

from torch import nn

ACTIVATION_FUNC = Literal["ReLU", "GELU"]


class FeedForward(nn.Module):
    def __init__(
        self, model_dim, dropout=0.1, activation_func: ACTIVATION_FUNC = "ReLU"
    ):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, model_dim * 4)
        if activation_func == "ReLU":
            self.activation_func = nn.ReLU()
        else:
            self.activation_func = nn.GELU()
        self.linear_2 = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        middle = self.linear_1(inputs)
        relu = self.activation_func(middle)
        output = self.linear_2(relu)
        # dropout
        output = self.dropout(output)
        return output
