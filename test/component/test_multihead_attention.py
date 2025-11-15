import torch

from component.multihead_attention import MultiheadAttention


class TestMultiheadAttention:
    def test_self_attention_output_shape(self):
        """Test self-attention output shape is correct"""
        batch_size = 3
        seq_len = 6
        model_dim = 512

        attention = MultiheadAttention(model_dim=model_dim)
        inputs = torch.randn(batch_size, seq_len, model_dim)
        output = attention(inputs)

        assert output.shape == (batch_size, seq_len, model_dim)
