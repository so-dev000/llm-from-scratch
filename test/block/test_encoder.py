import torch

from block.encoder import Encoder


class TestEncoder:
    def test_output_shape(self):
        batch_size = 3
        seq_len = 6
        model_dim = 512
        encoder_num = 6

        encoder = Encoder(model_dim=model_dim, encoder_num=encoder_num)
        inputs = torch.randn(batch_size, seq_len, model_dim)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, model_dim)
