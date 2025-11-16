import torch

from utils.collate import collate


class TestCollate:
    def test_padding_and_shape(self):
        batch = [
            {
                "src": torch.tensor([2, 10, 20, 3]),
                "tgt": torch.tensor([2, 30, 40, 3]),
                "src_text": "Hello",
                "tgt_text": "こんにちは",
            },
            {
                "src": torch.tensor([2, 50, 3]),
                "tgt": torch.tensor([2, 60, 70, 80, 3]),
                "src_text": "Hi",
                "tgt_text": "やあ",
            },
        ]

        result = collate(batch, pad_id=0)

        assert result["src"].shape == (2, 4)  # batch_size=2, max_len=4
        assert result["tgt"].shape == (2, 5)  # batch_size=2, max_len=5

        assert torch.all(result["src"][1] == torch.tensor([2, 50, 3, 0]))
        assert torch.all(result["tgt"][0] == torch.tensor([2, 30, 40, 3, 0]))

    def test_mask_generation(self):
        batch = [
            {
                "src": torch.tensor([2, 10, 20, 3]),
                "tgt": torch.tensor([2, 30, 3]),
                "src_text": "test",
                "tgt_text": "テスト",
            },
            {
                "src": torch.tensor([2, 50, 3]),
                "tgt": torch.tensor([2, 60, 70, 80, 3]),
                "src_text": "hi",
                "tgt_text": "やあ",
            },
        ]

        result = collate(batch, pad_id=0)

        expected_src_mask = torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        )
        expected_tgt_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )
        assert torch.all(result["src_mask"] == expected_src_mask)
        assert torch.all(result["tgt_mask"] == expected_tgt_mask)

    def test_single_sample_batch(self):
        batch = [
            {
                "src": torch.tensor([2, 10, 20, 3]),
                "tgt": torch.tensor([2, 30, 40, 50, 3]),
                "src_text": "test",
                "tgt_text": "テスト",
            }
        ]

        result = collate(batch, pad_id=0)

        assert result["src"].shape == (1, 4)
        assert result["tgt"].shape == (1, 5)
        assert torch.all(result["src_mask"])
        assert torch.all(result["tgt_mask"])
