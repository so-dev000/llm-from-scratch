import torch

from model.transformer import Transformer


class TestTransformer:
    def test_output_shape(self):
        """Test output shape is correct"""
        batch_size = 3
        seq_len = 6
        model_dim = 512
        encoder_num = 6
        decoder_num = 6

        transformer = Transformer(
            model_dim=model_dim, encoder_num=encoder_num, decoder_num=decoder_num
        )
        encoder_inputs = torch.randn(batch_size, seq_len, model_dim)
        decoder_inputs = torch.randn(batch_size, seq_len, model_dim)
        output = transformer(encoder_inputs, decoder_inputs)

        assert output.shape == (batch_size, seq_len, model_dim)
