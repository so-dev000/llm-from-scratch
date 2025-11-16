import torch


def create_causal_mask(seq_len, device=None):
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()


def combine_masks(padding_mask, causal_mask):
    # padding_mask: True is valid (batch_size, seq_len)
    # causal_mask: True is visible (seq_len, seq_len)

    # padding token can't see anything
    row_mask = padding_mask.unsqueeze(2)  # (batch, seq_len, 1)
    # padding token can't be seen
    col_mask = padding_mask.unsqueeze(1)  # (batch, 1, seq_len)
    causal = causal_mask.unsqueeze(0)  # (1, seq_len, seq_len)

    return row_mask & col_mask & causal
