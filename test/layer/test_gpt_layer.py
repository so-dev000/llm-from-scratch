import pytest
import torch

from layer.gpt_layer import GPTLayer


class TestGPTLayer:
    @pytest.fixture
    def gpt_layer(self):
        model_dim = 64
        num_heads = 4
        feedforward_dim = 128
        dropout = 0.1
        activation_func = "relu"
        return GPTLayer(model_dim, num_heads, feedforward_dim, dropout, activation_func)

    def test_forward_output_shape(self, gpt_layer):
        batch_size = 2
        seq_len = 10
        model_dim = 64

        inputs = torch.randn(batch_size, seq_len, model_dim)
        output = gpt_layer(inputs)

        assert output.shape == (batch_size, seq_len, model_dim)

    def test_forward_with_mask(self, gpt_layer):
        batch_size = 2
        seq_len = 10
        model_dim = 64

        inputs = torch.randn(batch_size, seq_len, model_dim)
        # Causal mask (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

        output = gpt_layer(inputs, mask=mask)
        assert output.shape == (batch_size, seq_len, model_dim)
