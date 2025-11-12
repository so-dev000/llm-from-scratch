import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, model_dim, middle_dim=3072, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, middle_dim)
        # cf: https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
        self.activation_func = nn.GELU()
        self.linear_2 = nn.Linear(middle_dim, model_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        middle = self.linear_1(inputs)
        gelu = self.activation_func(middle)
        output = self.linear_2(gelu)
        # dropout
        output = self.dropout(output)
        return output


if __name__ == "__main__":
    batch_size = 3
    seq_len = 6
    model_dim = 512

    input = torch.randn(batch_size, seq_len, model_dim)
    model = FeedForward(model_dim=model_dim)

    print(f"Input shape: {input.shape}")  # (3, 6, 512)
    output = model.forward(input)
    print(f"Output shape: {output.shape}")  # (3, 6, 512)
