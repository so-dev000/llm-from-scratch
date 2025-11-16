import torch

from model.transformer import Transformer


class TestTransformer:
    def test_output_shape(self):
        vocab_size = 10000
        batch_size = 3
        seq_len = 6
        model_dim = 512
        encoder_num = 6
        decoder_num = 6

        transformer = Transformer(
            vocab_size=vocab_size,
            model_dim=model_dim,
            encoder_num=encoder_num,
            decoder_num=decoder_num,
        )
        encoder_inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        decoder_inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = transformer(encoder_inputs, decoder_inputs)

        assert output.shape == (batch_size, seq_len, vocab_size)
