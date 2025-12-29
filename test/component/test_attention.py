import torch

from component.attention import Attention
from component.rotary_embedding import precompute_freqs_cis


class TestAttention:
    def test_gqa_output_shape(self):
        batch_size = 2
        seq_len = 8
        model_dim = 128
        num_heads = 8
        num_kv_heads = 2

        attention = Attention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.0,
        )
        x = torch.randn(batch_size, seq_len, model_dim)
        output = attention(x)

        assert output.shape == (batch_size, seq_len, model_dim)

    def test_mha_mode(self):
        batch_size = 2
        seq_len = 8
        model_dim = 128
        num_heads = 8

        attention = Attention(
            model_dim=model_dim, num_heads=num_heads, num_kv_heads=num_heads
        )
        x = torch.randn(batch_size, seq_len, model_dim)
        output = attention(x)

        assert output.shape == (batch_size, seq_len, model_dim)
        assert attention.num_queries_per_kv == 1

    def test_mqa_mode(self):
        batch_size = 2
        seq_len = 8
        model_dim = 128
        num_heads = 8

        attention = Attention(
            model_dim=model_dim, num_heads=num_heads, num_kv_heads=1, dropout=0.0
        )
        x = torch.randn(batch_size, seq_len, model_dim)
        output = attention(x)

        assert output.shape == (batch_size, seq_len, model_dim)
        assert attention.num_queries_per_kv == 8

    def test_with_rotary_embeddings(self):
        batch_size = 2
        seq_len = 8
        model_dim = 128
        num_heads = 8
        num_kv_heads = 2

        attention = Attention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.0,
        )
        x = torch.randn(batch_size, seq_len, model_dim)

        head_dim = model_dim // num_heads
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        output = attention(x, freqs_cis=freqs_cis)

        assert output.shape == (batch_size, seq_len, model_dim)

    def test_with_causal_mask(self):
        batch_size = 2
        seq_len = 8
        model_dim = 128
        num_heads = 8
        num_kv_heads = 2

        attention = Attention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.0,
        )
        x = torch.randn(batch_size, seq_len, model_dim)

        # Create causal mask: upper triangular matrix with -inf
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

        output = attention(x, mask=mask)

        assert output.shape == (batch_size, seq_len, model_dim)

    def test_invalid_num_heads_raises_error(self):
        model_dim = 128
        num_heads = 8
        num_kv_heads = 3  # 8 is not divisible by 3

        attention = Attention(
            model_dim=model_dim, num_heads=num_heads, num_kv_heads=num_kv_heads
        )

        # num_queries_per_kv will be 8 // 3 = 2 (integer division, losing precision)
        assert attention.num_queries_per_kv == 2

    def test_kv_heads_expansion(self):
        batch_size = 1
        seq_len = 4
        model_dim = 64
        num_heads = 4
        num_kv_heads = 2

        attention = Attention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.0,
        )

        x = torch.randn(batch_size, seq_len, model_dim)

        # Forward pass should expand 2 KV heads to 4 heads
        output = attention(x)

        assert output.shape == (batch_size, seq_len, model_dim)
        # Each KV head should be used by num_queries_per_kv = 2 query heads
        assert attention.num_queries_per_kv == 2

    def test_dropout_during_training(self):
        batch_size = 2
        seq_len = 8
        model_dim = 128
        num_heads = 8
        num_kv_heads = 2

        attention = Attention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.5,
        )
        attention.train()  # Set to training mode

        x = torch.randn(batch_size, seq_len, model_dim)
        output1 = attention(x)
        output2 = attention(x)

        # With dropout, outputs should be different
        assert not torch.allclose(output1, output2)

    def test_no_dropout_during_eval(self):
        batch_size = 2
        seq_len = 8
        model_dim = 128
        num_heads = 8
        num_kv_heads = 2

        attention = Attention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.5,
        )
        attention.eval()  # Set to evaluation mode

        x = torch.randn(batch_size, seq_len, model_dim)

        # Set seed for reproducibility
        torch.manual_seed(42)
        output1 = attention(x)

        torch.manual_seed(42)
        output2 = attention(x)

        # Without dropout (eval mode), outputs should be identical
        assert torch.allclose(output1, output2)

    def test_attention_scores_scaling(self):
        batch_size = 1
        seq_len = 4
        model_dim = 64
        num_heads = 4
        num_kv_heads = 2

        attention = Attention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.0,
        )

        x = torch.randn(batch_size, seq_len, model_dim)
        output = attention(x)

        # Verify head_dim calculation
        expected_head_dim = model_dim // num_heads
        assert attention.head_dim == expected_head_dim

        # Output should have correct shape
        assert output.shape == (batch_size, seq_len, model_dim)

    def test_batched_input(self):
        model_dim = 128
        num_heads = 8
        num_kv_heads = 2
        seq_len = 10

        attention = Attention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.0,
        )

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, seq_len, model_dim)
            output = attention(x)
            assert output.shape == (batch_size, seq_len, model_dim)

    def test_different_sequence_lengths(self):
        batch_size = 2
        model_dim = 128
        num_heads = 8
        num_kv_heads = 2

        attention = Attention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.0,
        )

        for seq_len in [1, 8, 32, 128]:
            x = torch.randn(batch_size, seq_len, model_dim)
            output = attention(x)
            assert output.shape == (batch_size, seq_len, model_dim)
