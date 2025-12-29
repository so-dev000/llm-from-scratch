import torch
from torch import nn
from torch.nn import functional as F


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, model_dim: int, feedforward_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(model_dim, feedforward_dim, bias=False)
        self.w2 = nn.Linear(model_dim, feedforward_dim, bias=False)
        self.w3 = nn.Linear(feedforward_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cf: https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        out = self.w3(gate * up)
        return self.dropout(out)
