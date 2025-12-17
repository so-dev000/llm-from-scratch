import pytest
import torch

from block.gpt_block import GPTBlock


class TestGPTBlock:
    @pytest.fixture
    def gpt_block(self):
        model_dim = 64
        layer_num = 2
        num_heads = 4
        feedforward_dim = 128
        dropout = 0.1
        activation_func = "relu"
        return GPTBlock(
            model_dim, layer_num, num_heads, feedforward_dim, dropout, activation_func
        )

    def test_forward_output_shape(self, gpt_block):
        batch_size = 2
        seq_len = 10
        model_dim = 64

        inputs = torch.randn(batch_size, seq_len, model_dim)
        output = gpt_block(inputs)

        assert output.shape == (batch_size, seq_len, model_dim)

    def test_forward_with_mask(self, gpt_block):
        batch_size = 2
        seq_len = 10
        model_dim = 64

        inputs = torch.randn(batch_size, seq_len, model_dim)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

        output = gpt_block(inputs, mask=mask)
        assert output.shape == (batch_size, seq_len, model_dim)
