import os
from typing import Optional

import pytorch_lightning as L
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader

from tokenizer.bpe import BPE
from utils.collate import collate


class TransformerLRScheduler(L.Callback):
    def __init__(self, model_dim: int, warmup_steps: int):
        super().__init__()
        self.model_dim = model_dim
        self.warmup_steps = warmup_steps
        self.step = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.step += 1
        lr = self._get_lr(self.step)

        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        pl_module.log("learning_rate", lr, on_step=True, on_epoch=False)

    def _get_lr(self, step):
        step = max(step, 1)
        lr = (self.model_dim**-0.5) * min(step**-0.5, step * self.warmup_steps**-1.5)
        return lr


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
        tokenizer_dir = self.config.tokenizer_dir + "/bsd_en_ja"
        en_tokenizer_path = f"{tokenizer_dir}/en_bpe.pkl"
        ja_tokenizer_path = f"{tokenizer_dir}/ja_bpe.pkl"

        if not os.path.exists(en_tokenizer_path) or not os.path.exists(
            ja_tokenizer_path
        ):
            raise FileNotFoundError(
                "Tokenizers not found. Run scripts/prepare.py first"
            )

        self.src_tokenizer = BPE.load(en_tokenizer_path)
        self.tgt_tokenizer = BPE.load(ja_tokenizer_path)

        self.config.model.src_vocab_size = self.src_tokenizer.get_vocab_size()
        self.config.model.tgt_vocab_size = self.tgt_tokenizer.get_vocab_size()

        dataset = load_dataset(self.config.data.dataset_name, split="train")
        train_rest = dataset.train_test_split(test_size=0.15, seed=42)
        val_test = train_rest["test"].train_test_split(test_size=0.33, seed=42)
        train_val = DatasetDict(
            {"train": train_rest["train"], "val": val_test["train"]}
        )

        def preprocess_batch(batch):
            src_ids = []
            for text in batch["en_sentence"]:
                ids = self.src_tokenizer.encode(text, add_special_tokens=True)
                if len(ids) > self.config.data.max_length:
                    ids = ids[: self.config.data.max_length - 1] + [
                        self.src_tokenizer.special_tokens["<EOS>"]
                    ]
                src_ids.append(ids)

            tgt_ids = []
            for text in batch["ja_sentence"]:
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
            batch_size=1000,
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
        self.tokenized_datasets = None
        self.tokenizer = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError("GPTDataModule not yet implemented")

    def train_dataloader(self):
        raise NotImplementedError("GPTDataModule not yet implemented")

    def val_dataloader(self):
        raise NotImplementedError("GPTDataModule not yet implemented")


def get_data_module(config) -> L.LightningDataModule:
    if config.model.model_type == "transformer":
        return TransformerDataModule(config)
    elif config.model.model_type == "gpt":
        return GPTDataModule(config)
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")
