import torch

from layer.encoder_layer import EncoderLayer


class TestEncoderLayer:
    def test_output_shape(self):
        batch_size = 3
        seq_len = 6
        model_dim = 512

        encoder_layer = EncoderLayer(model_dim=model_dim)
        inputs = torch.randn(batch_size, seq_len, model_dim)
        output = encoder_layer(inputs)

        assert output.shape == (batch_size, seq_len, model_dim)
