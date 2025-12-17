import torch

from model.transformer import Transformer
from scripts.config import TransformerModelConfig


class TestTransformer:
    def test_output_shape(self):
        src_vocab_size = 10000
        tgt_vocab_size = 12000
        batch_size = 3
        seq_len = 6
        model_dim = 512
        encoder_num = 6
        decoder_num = 6

        config = TransformerModelConfig(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            model_dim=model_dim,
            encoder_layers=encoder_num,
            decoder_layers=decoder_num,
            num_heads=8,
            feedforward_dim=model_dim * 4,
            dropout=0.1,
            activation="relu",
            max_seq_len=5000,
            padding_idx=0,
        )

        transformer = Transformer(config)
        encoder_inputs = torch.randint(0, src_vocab_size, (batch_size, seq_len))
        decoder_inputs = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))
        output = transformer(encoder_inputs, decoder_inputs)

        assert output.shape == (batch_size, seq_len, tgt_vocab_size)
