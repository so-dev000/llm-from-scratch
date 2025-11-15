import pytest
import torch

from component.multihead_attention import MultiheadAttention


class TestMultiheadAttention:
    def test_self_attention_output_shape(self):
        batch_size = 3
        seq_len = 6
        model_dim = 512

        attention = MultiheadAttention(model_dim=model_dim)
        inputs = torch.randn(batch_size, seq_len, model_dim)
        output = attention(inputs)

        assert output.shape == (batch_size, seq_len, model_dim)

    def test_invalid_model_dim_raises_value_error(self):
        model_dim = 513  # Not divisible by default head=8
        with pytest.raises(ValueError, match="model_dim must be divisible by head"):
            MultiheadAttention(model_dim=model_dim)
