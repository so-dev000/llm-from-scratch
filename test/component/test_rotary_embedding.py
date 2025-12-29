import torch

from component.rotary_embedding import (
    RotaryEmbedding,
    apply_rotary_emb,
    precompute_freqs_cis,
)


class TestPrecomputeFreqsCis:
    def test_output_shape(self):
        dim = 64
        seq_len = 100
        freqs_cis = precompute_freqs_cis(dim, seq_len)

        assert freqs_cis.shape == (seq_len, dim // 2)
        assert freqs_cis.dtype == torch.complex64

    def test_position_zero_is_identity(self):
        dim = 64
        freqs_cis = precompute_freqs_cis(dim, seq_len=10)

        # Position 0 should be e^(i*0) = 1+0j
        expected = torch.ones(dim // 2, dtype=torch.complex64)
        assert torch.allclose(freqs_cis[0], expected)

    def test_magnitude_is_one(self):
        dim = 64
        seq_len = 50
        freqs_cis = precompute_freqs_cis(dim, seq_len)

        # All complex numbers should have magnitude 1
        magnitudes = freqs_cis.abs()
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes))


class TestApplyRotaryEmb:
    def test_output_shape(self):
        batch_size = 2
        seq_len = 10
        n_heads = 8
        head_dim = 64

        xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
        xk = torch.randn(batch_size, seq_len, n_heads, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

        assert xq_out.shape == xq.shape
        assert xk_out.shape == xk.shape

    def test_dtype_preservation(self):
        batch_size = 2
        seq_len = 10
        n_heads = 4
        head_dim = 64

        for dtype in [torch.float32, torch.bfloat16]:
            xq = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype)
            xk = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype)
            freqs_cis = precompute_freqs_cis(head_dim, seq_len)

            xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

            assert xq_out.dtype == dtype
            assert xk_out.dtype == dtype

    def test_different_positions_different_values(self):
        batch_size = 1
        seq_len = 10
        n_heads = 4
        head_dim = 64

        xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
        xk = torch.randn(batch_size, seq_len, n_heads, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

        # Different positions should have different rotations
        assert not torch.allclose(xq_out[0, 0], xq_out[0, 1], atol=1e-3)


class TestRotaryEmbedding:
    def test_forward_output_shape(self):
        dim = 64
        max_seq_len = 2048
        rope = RotaryEmbedding(dim=dim, max_seq_len=max_seq_len)

        batch_size = 2
        seq_len = 10
        n_heads = 8

        xq = torch.randn(batch_size, seq_len, n_heads, dim)
        xk = torch.randn(batch_size, seq_len, n_heads, dim)

        xq_out, xk_out = rope(xq, xk)

        assert xq_out.shape == xq.shape
        assert xk_out.shape == xk.shape

    def test_incremental_decoding(self):
        dim = 64
        rope = RotaryEmbedding(dim=dim, max_seq_len=100)

        batch_size = 1
        n_heads = 4

        # Simulate incremental decoding
        # First 5 tokens
        xq1 = torch.randn(batch_size, 5, n_heads, dim)
        xk1 = torch.randn(batch_size, 5, n_heads, dim)
        xq_out1, xk_out1 = rope(xq1, xk1, start_pos=0)

        # Next 1 token (position 5)
        xq2 = torch.randn(batch_size, 1, n_heads, dim)
        xk2 = torch.randn(batch_size, 1, n_heads, dim)
        xq_out2, xk_out2 = rope(xq2, xk2, start_pos=5)

        assert xq_out2.shape == (batch_size, 1, n_heads, dim)

    def test_buffer_registration(self):
        dim = 64
        rope = RotaryEmbedding(dim=dim)

        # freqs_cis should be registered as buffer
        assert "freqs_cis" in rope._buffers
        assert rope.freqs_cis is not None
