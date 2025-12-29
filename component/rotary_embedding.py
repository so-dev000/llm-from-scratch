import torch
from torch import nn


def precompute_freqs_cis(
    dim: int, seq_len: int, theta: float = 10000.0, device: str = "cpu"
) -> torch.Tensor:
    # Compute frequencies: θⱼ = 10000^(-2j/d)
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )

    # Create position index: [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, device=device, dtype=torch.float32)

    # Compute m·θⱼ
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i·m·θⱼ) = cos(m·θⱼ) + i·sin(m·θⱼ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Convert to complex and reshape
    # (batch, seq, heads, dim) → (batch, seq, heads, dim/2, 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Broadcast
    freqs_cis = freqs_cis[None, :, None, :]

    # Apply rotation
    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis

    # Convert to real numbers
    # (batch, seq, heads, dim/2, 2) → (batch, seq, heads, dim)
    xq_out = torch.view_as_real(xq_out).flatten(3)
    xk_out = torch.view_as_real(xk_out).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # xq: (batch, seq, heads, dim)
        seq_len = xq.shape[1]
        # Slice freqs_cis
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        return apply_rotary_emb(xq, xk, freqs_cis)
