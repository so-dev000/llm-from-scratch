import torch

from component.token_embedding import TokenEmbedding


class TestTokenEmbedding:
    def test_output_shape(self):
        """Test output shape is correct"""
        vocab_size = 30000
        model_dim = 512
        batch_size = 2
        seq_len = 10

        embedding = TokenEmbedding(vocab_size=vocab_size, model_dim=model_dim)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = embedding(token_ids)

        assert output.shape == (batch_size, seq_len, model_dim)
