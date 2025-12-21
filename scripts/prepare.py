import os
from itertools import islice

import modal
from config import Config
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tqdm import tqdm

app = modal.App("llm-data-preparation")

volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("regex", "tqdm", "datasets", "tokenizers")
    .add_local_dir("tokenizer", remote_path="/root/llm-from-scratch/tokenizer")
    .add_local_file("scripts/config.py", remote_path="/root/config.py")
)

GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?"
    r"[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def prepare_transformer_tokenizers(
    dataset_name,
    vocab_size,
    src_lang,
    tgt_lang,
    src_column,
    tgt_column,
):
    from tokenizer.bpe import BPE

    dataset_dir = dataset_name.replace("/", "_")
    tokenizer_dir = f"data/tokenizers/{dataset_dir}"
    src_tokenizer_path = f"{tokenizer_dir}/{src_lang}_bpe.pkl"
    tgt_tokenizer_path = f"{tokenizer_dir}/{tgt_lang}_bpe.pkl"
    dataset = load_dataset(dataset_name, split="train")
    src_texts = [
        ex[src_column] for ex in tqdm(dataset, desc=f"Loading {src_lang} texts")
    ]
    src_tokenizer = BPE(pattern=GPT2_PATTERN if src_lang == "en" else None)
    src_tokenizer.train(src_texts, vocab_size=vocab_size)
    tgt_texts = [
        ex[tgt_column] for ex in tqdm(dataset, desc=f"Loading {tgt_lang} texts")
    ]
    tgt_tokenizer = BPE(pattern=None)
    tgt_tokenizer.train(tgt_texts, vocab_size=vocab_size)
    os.makedirs(tokenizer_dir, exist_ok=True)
    src_tokenizer.save(src_tokenizer_path)
    tgt_tokenizer.save(tgt_tokenizer_path)
    return tokenizer_dir


@app.function(image=image, volumes={"/vol": volume}, timeout=3600)
def prepare_gpt_tokenizer(
    dataset_name,
    dataset_config,
    vocab_size,
    text_column,
):
    dataset_dir = dataset_name.replace("/", "_")

    tokenizer_dir = f"/vol/tokenizers/{dataset_dir}"
    tokenizer_path = f"{tokenizer_dir}/tokenizer.json"

    num_samples = 1_000_000
    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    def text_iterator():
        for example in tqdm(
            islice(dataset, num_samples), total=num_samples, desc="Loading texts"
        ):
            yield example[text_column]

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        show_progress=False,
    )
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(tokenizer_path)
    volume.commit()
    return tokenizer_dir


@app.local_entrypoint()
def main(
    model_type: str = "transformer",
    dataset: str = None,
    vocab_size: int = None,
    dataset_config: str = None,
):
    if model_type == "transformer":
        config = Config.for_transformer()
    elif model_type == "gpt":
        config = Config.for_gpt()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    dataset_name = dataset or config.data.dataset_name
    vocab_size_value = vocab_size or config.data.vocab_size

    if model_type == "transformer":
        prepare_transformer_tokenizers(
            dataset_name=dataset_name,
            vocab_size=vocab_size_value,
            src_lang=config.data.src_lang,
            tgt_lang=config.data.tgt_lang,
            src_column=config.data.src_column,
            tgt_column=config.data.tgt_column,
        )
    elif model_type == "gpt":
        dataset_config_value = dataset_config or config.data.dataset_config
        prepare_gpt_tokenizer.remote(
            dataset_name=dataset_name,
            dataset_config=dataset_config_value,
            vocab_size=vocab_size_value,
            text_column=config.data.text_column,
        )
