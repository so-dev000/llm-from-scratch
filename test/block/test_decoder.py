import torch

from block.decoder import Decoder


class TestDecoder:
    def test_output_shape(self):
        """Test output shape is correct"""
        batch_size = 3
        seq_len = 6
        model_dim = 512
        decoder_num = 6

        decoder = Decoder(model_dim=model_dim, decoder_num=decoder_num)
        inputs = torch.randn(batch_size, seq_len, model_dim)
        encoder_out = torch.randn(batch_size, seq_len, model_dim)
        output = decoder(inputs, encoder_out)

        assert output.shape == (batch_size, seq_len, model_dim)
