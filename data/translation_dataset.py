import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, data, en_tokenizer, ja_tokenizer, max_length=128):
        self.data = data
        self.en_tokenizer = en_tokenizer
        self.ja_tokenizer = ja_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        en_text = item["en_sentence"]
        ja_text = item["ja_sentence"]

        # tokenize
        src_ids = self.en_tokenizer.encode(en_text, add_special_tokens=True)
        tgt_ids = self.ja_tokenizer.encode(ja_text, add_special_tokens=True)

        # truncate
        src_ids = src_ids[: self.max_length]
        tgt_ids = tgt_ids[: self.max_length]

        # convert to tensor
        src_tensor = torch.tensor(src_ids, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)

        return {
            "src": src_tensor,
            "tgt": tgt_tensor,
            "src_text": en_text,
            "tgt_text": ja_text,
        }
