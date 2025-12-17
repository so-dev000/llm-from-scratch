import torch

from layer.decoder_layer import DecoderLayer


class TestDecoderLayer:
    def test_output_shape(self):
        batch_size = 3
        seq_len = 6
        model_dim = 512

        decoder_layer = DecoderLayer(
            model_dim=model_dim,
            num_heads=8,
            feedforward_dim=model_dim * 4,
            dropout=0.1,
            activation_func="ReLU",
        )
        inputs = torch.randn(batch_size, seq_len, model_dim)
        encoder_out = torch.randn(batch_size, seq_len, model_dim)
        output = decoder_layer(inputs, encoder_out)

        assert output.shape == (batch_size, seq_len, model_dim)
