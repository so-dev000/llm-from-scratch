import torch

from component.feed_forward import FeedForward


class TestFeedForward:
    def test_output_shape(self):
        batch_size = 3
        seq_len = 6
        model_dim = 512

        ff = FeedForward(model_dim=model_dim)
        inputs = torch.randn(batch_size, seq_len, model_dim)
        output = ff(inputs)

        assert output.shape == (batch_size, seq_len, model_dim)
