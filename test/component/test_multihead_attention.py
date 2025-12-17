import pytest
import torch

from component.multihead_attention import MultiheadAttention


class TestMultiheadAttention:
    def test_self_attention_output_shape(self):
        batch_size = 4
        seq_len = 10
        model_dim = 64

        attention = MultiheadAttention(model_dim=model_dim, num_heads=8, dropout=0.1)
        inputs = torch.randn(batch_size, seq_len, model_dim)
        output = attention(inputs)

        assert output.shape == (batch_size, seq_len, model_dim)

    def test_invalid_model_dim_raises_value_error(self):
        model_dim = 65  # Not divisible by default num_heads=8
        with pytest.raises(
            ValueError, match="model_dim must be divisible by num_heads"
        ):
            MultiheadAttention(model_dim=model_dim, num_heads=8, dropout=0.1)

    def test_with_2d_mask(self):
        model_dim = 64
        seq_len = 5
        attention = MultiheadAttention(model_dim=model_dim, num_heads=8, dropout=0.1)
        inputs = torch.randn(2, seq_len, model_dim)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()  # (seq, seq)
        output = attention(inputs, mask=mask)
        assert output.shape == inputs.shape
        assert attention.last_attention_weights is not None

    def test_with_3d_mask(self):
        batch_size = 2
        seq_len = 4
        model_dim = 64

        attention = MultiheadAttention(model_dim=model_dim, num_heads=8, dropout=0.1)
        inputs = torch.randn(batch_size, seq_len, model_dim)
        mask = torch.tril(torch.ones(batch_size, seq_len, seq_len)).bool()  # (2, 4, 4)

        output = attention(inputs, mask=mask)

        assert output.shape == (batch_size, seq_len, model_dim)

    def test_cross_attention_with_padding_mask(self):
        batch_size = 2
        tgt_seq_len = 4
        src_seq_len = 5
        model_dim = 512

        attention = MultiheadAttention(model_dim=model_dim, num_heads=8, dropout=0.1)
        query_inputs = torch.randn(batch_size, tgt_seq_len, model_dim)
        encoder_outputs = torch.randn(batch_size, src_seq_len, model_dim)
        mask = torch.ones(batch_size, src_seq_len).bool()
        mask[:, -2:] = False  # Mask last two tokens of source

        output = attention(query_inputs, encoder_out=encoder_outputs, mask=mask)

        assert output.shape == (batch_size, tgt_seq_len, model_dim)
