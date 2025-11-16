import torch

from utils.masking import combine_masks, create_causal_mask


class TestMasking:
    def test_causal_mask(self):
        mask = create_causal_mask(4)

        expected = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )

        assert mask.shape == (4, 4)
        assert torch.all(mask == expected)

    def test_combine_masks(self):
        padding_mask = torch.tensor([[True, True, False]])
        causal_mask = create_causal_mask(3)

        combined = combine_masks(padding_mask, causal_mask)

        expected = torch.tensor(
            [[[True, False, False], [True, True, False], [False, False, False]]]
        )

        assert combined.shape == (1, 3, 3)
        assert torch.all(combined == expected)
