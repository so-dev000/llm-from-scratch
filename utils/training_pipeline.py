import os
from typing import Optional

import pytorch_lightning as L
from datasets import DatasetDict, load_dataset, load_from_disk
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from tokenizer.bpe import BPE
from utils.collate import collate


class TransformerDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenized_datasets = None
        self.src_tokenizer = None
        self.tgt_tokenizer = None

    def prepare_data(self):
        load_dataset(self.config.data.dataset_name, split="train")

    def setup(self, stage: Optional[str] = None):
        if self.tokenized_datasets is not None:
            return

        dataset_dir = self.config.data.dataset_name.replace("/", "_")
        tokenizer_dir = f"{self.config.tokenizer_dir}/{dataset_dir}"

        src_lang = self.config.data.src_lang
        tgt_lang = self.config.data.tgt_lang

        src_tokenizer_path = f"{tokenizer_dir}/{src_lang}_bpe.pkl"
        tgt_tokenizer_path = f"{tokenizer_dir}/{tgt_lang}_bpe.pkl"

        if not os.path.exists(src_tokenizer_path) or not os.path.exists(
            tgt_tokenizer_path
        ):
            raise FileNotFoundError(
                f"Tokenizers not found at {tokenizer_dir}. Run scripts/prepare.py first"
            )

        self.src_tokenizer = BPE.load(src_tokenizer_path)
        self.tgt_tokenizer = BPE.load(tgt_tokenizer_path)

        self.config.model.src_vocab_size = self.src_tokenizer.get_vocab_size()
        self.config.model.tgt_vocab_size = self.tgt_tokenizer.get_vocab_size()

        dataset = load_dataset(self.config.data.dataset_name, split="train")
        split_data = dataset.train_test_split(
            test_size=self.config.data.val_split_size, seed=42
        )
        train_val = DatasetDict(
            {"train": split_data["train"], "val": split_data["test"]}
        )
        src_column = self.config.data.src_column
        tgt_column = self.config.data.tgt_column

        def preprocess_batch(batch):
            src_ids = []
            for text in batch[src_column]:
                ids = self.src_tokenizer.encode(text, add_special_tokens=True)
                if len(ids) > self.config.data.max_length:
                    ids = ids[: self.config.data.max_length - 1] + [
                        self.src_tokenizer.special_tokens["<EOS>"]
                    ]
                src_ids.append(ids)
            tgt_ids = []
            for text in batch[tgt_column]:
                ids = self.tgt_tokenizer.encode(text, add_special_tokens=True)
                if len(ids) > self.config.data.max_length:
                    ids = ids[: self.config.data.max_length - 1] + [
                        self.tgt_tokenizer.special_tokens["<EOS>"]
                    ]
                tgt_ids.append(ids)
            return {"src": src_ids, "tgt": tgt_ids}

        self.tokenized_datasets = train_val.map(
            preprocess_batch,
            batched=True,
            batch_size=self.config.data.preprocess_batch_size,
            remove_columns=train_val["train"].column_names,
            desc="Tokenizing dataset",
        )

        self.tokenized_datasets.set_format(type="torch", columns=["src", "tgt"])

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["train"],
            batch_size=self.config.data.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.persistent_workers,
            prefetch_factor=self.config.data.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["val"],
            batch_size=self.config.data.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.persistent_workers,
            prefetch_factor=self.config.data.prefetch_factor,
        )


class GPTDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        dataset_dir = self.config.data.dataset_name.replace("/", "_")
        tokenizer_path = f"{self.config.tokenizer_dir}/{dataset_dir}/tokenizer.json"

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.config.model.vocab_size = self.tokenizer.get_vocab_size()

        num_samples = self.config.data.tokenizer_train_samples

        dataset = load_dataset(
            self.config.data.dataset_name,
            self.config.data.dataset_config,
            split="train",
            streaming=False,
        )

        dataset = dataset.select(range(min(num_samples, len(dataset))))

        split_data = dataset.train_test_split(
            test_size=self.config.data.val_split_size, seed=42
        )

        def preprocess_batch(batch):
            texts = batch[self.config.data.text_column]
            encodings = self.tokenizer.encode_batch(texts)
            max_len = self.config.data.max_length
            token_ids = [
                enc.ids[:max_len] if len(enc.ids) > max_len else enc.ids
                for enc in encodings
            ]
            return {"input_ids": token_ids}

        train_data = split_data["train"].map(
            preprocess_batch,
            batched=True,
            batch_size=self.config.data.preprocess_batch_size,
            num_proc=self.config.data.preprocess_num_proc,
            remove_columns=split_data["train"].column_names,
            desc="Tokenizing train data",
        )
        val_data = split_data["test"].map(
            preprocess_batch,
            batched=True,
            batch_size=self.config.data.preprocess_batch_size,
            num_proc=self.config.data.preprocess_num_proc,
            remove_columns=split_data["test"].column_names,
            desc="Tokenizing val data",
        )

        train_data.set_format(type="torch", columns=["input_ids"])
        val_data.set_format(type="torch", columns=["input_ids"])

        self.train_dataset = train_data
        self.val_dataset = val_data

    def train_dataloader(self):
        from utils.collate import collate_gpt

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            collate_fn=collate_gpt,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.persistent_workers,
            prefetch_factor=self.config.data.prefetch_factor,
        )

    def val_dataloader(self):
        from utils.collate import collate_gpt

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            collate_fn=collate_gpt,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.persistent_workers,
            prefetch_factor=self.config.data.prefetch_factor,
        )


class LlamaDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        dataset_dir = self.config.data.dataset_name.replace("/", "_")
        tokenizer_path = f"{self.config.tokenizer_dir}/{dataset_dir}/tokenizer.json"

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.config.model.vocab_size = self.tokenizer.get_vocab_size()

        # Load preprocessed data from disk/volume
        preprocessed_dir = f"/vol/preprocessed/{dataset_dir}"
        if not os.path.exists(preprocessed_dir):
            raise FileNotFoundError(
                f"Preprocessed data not found at {preprocessed_dir}. "
                f"Run scripts/preprocess_local.py first and upload to Modal volume."
            )

        datasets = load_from_disk(preprocessed_dir)
        datasets.set_format(type="torch", columns=["input_ids"])

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["val"]

    def train_dataloader(self):
        from utils.collate import collate_gpt

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            collate_fn=collate_gpt,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.persistent_workers,
            prefetch_factor=self.config.data.prefetch_factor,
        )

    def val_dataloader(self):
        from utils.collate import collate_gpt

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            collate_fn=collate_gpt,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.persistent_workers,
            prefetch_factor=self.config.data.prefetch_factor,
        )


def get_data_module(config) -> L.LightningDataModule:
    if config.model.model_type == "transformer":
        return TransformerDataModule(config)
    elif config.model.model_type == "gpt":
        return GPTDataModule(config)
    elif config.model.model_type == "llama":
        return LlamaDataModule(config)
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")
