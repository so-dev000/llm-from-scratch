from torch import nn


class FeedForward(nn.Module):
    def __init__(self, model_dim, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, model_dim * 4)
        self.activation_func = nn.ReLU()
        self.linear_2 = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        middle = self.linear_1(inputs)
        relu = self.activation_func(middle)
        output = self.linear_2(relu)
        # dropout
        output = self.dropout(output)
        return output
