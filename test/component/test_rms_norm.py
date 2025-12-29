import torch

from component.rms_norm import RMSNorm


class TestRMSNorm:
    def test_output_shape(self):
        batch_size = 2
        seq_len = 10
        dim = 64

        rms_norm = RMSNorm(dim=dim)
        x = torch.randn(batch_size, seq_len, dim)
        output = rms_norm(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_normalized_output(self):
        dim = 64
        rms_norm = RMSNorm(dim=dim)
        x = torch.randn(10, dim)
        output = rms_norm(x)

        # Check that RMS is approximately 1.0 after normalization
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt((output / rms_norm.weight).pow(2).mean(-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_learnable_weight(self):
        dim = 64
        rms_norm = RMSNorm(dim=dim)

        # Weight should be learnable parameter
        assert rms_norm.weight.requires_grad
        assert rms_norm.weight.shape == (dim,)

    def test_dtype_preservation(self):
        dim = 64
        rms_norm = RMSNorm(dim=dim)

        # Test with float32
        x = torch.randn(2, 10, dim, dtype=torch.float32)
        output = rms_norm(x)
        assert output.dtype == torch.float32

    def test_zero_mean_not_required(self):
        # RMSNorm doesn't require zero mean
        dim = 64
        rms_norm = RMSNorm(dim=dim)

        # Create input with non-zero mean
        x = torch.randn(2, 10, dim) + 5.0
        output = rms_norm(x)

        # Should still work and produce valid output
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
