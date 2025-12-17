import torch

from component.positional_encoding import PositionalEncoding


class TestPositionalEncoding:
    def test_output_shape(self):
        model_dim = 512
        batch_size = 2
        seq_len = 10

        pe = PositionalEncoding(model_dim=model_dim, dropout=0.0, max_len=5000)
        inputs = torch.randn(batch_size, seq_len, model_dim)
        output = pe(inputs)

        assert output.shape == (batch_size, seq_len, model_dim)
