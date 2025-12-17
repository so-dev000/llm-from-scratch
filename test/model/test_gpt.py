import pytest
import torch

from model.gpt import GPT
from scripts.config import GPTModelConfig


class TestGPT:
    @pytest.fixture
    def config(self):
        return GPTModelConfig(
            vocab_size=1000,
            max_seq_len=20,
            model_dim=64,
            num_heads=4,
            num_layers=2,
            feedforward_dim=128,
            dropout=0.1,
            activation="relu",
            padding_idx=0,
        )

    def test_forward_output_shape(self, config):
        batch_size = 2
        seq_len = 10
        model = GPT(config)

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        output = model(input_ids)

        assert output.shape == (batch_size, seq_len, config.vocab_size)

    def test_inference_methods_stubs(self, config):
        model = GPT(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with pytest.raises(NotImplementedError):
            model.prepare_context(input_ids)

        with pytest.raises(NotImplementedError):
            model.generate_next_token(input_ids, None)

    def test_causal_masking_logic(self, config):
        model = GPT(config)
        model.eval()

        torch.manual_seed(42)
        t1, t2, t3, t4 = 10, 20, 30, 40
        seq_a = torch.tensor([[t1, t2, t3]])
        seq_b = torch.tensor([[t1, t2, t4]])

        with torch.no_grad():
            out_a = model(seq_a)
            out_b = model(seq_b)

        assert torch.allclose(out_a[0, 1], out_b[0, 1], atol=1e-3)
        assert not torch.allclose(out_a[0, 2], out_b[0, 2], atol=1e-5)
