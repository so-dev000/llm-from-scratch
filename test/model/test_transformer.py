import torch

from model.transformer import Transformer


class TestTransformer:
    def test_output_shape(self):
        src_vocab_size = 10000
        tgt_vocab_size = 12000
        batch_size = 3
        seq_len = 6
        model_dim = 512
        encoder_num = 6
        decoder_num = 6

        transformer = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            model_dim=model_dim,
            encoder_num=encoder_num,
            decoder_num=decoder_num,
        )
        encoder_inputs = torch.randint(0, src_vocab_size, (batch_size, seq_len))
        decoder_inputs = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))
        output = transformer(encoder_inputs, decoder_inputs)

        assert output.shape == (batch_size, seq_len, tgt_vocab_size)
