import os

import modal

app = modal.App("llm-data-preparation")

volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

upload_image = modal.Image.debian_slim(python_version="3.11").pip_install("tqdm")

GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?"
    r"[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def train_transformer_tokenizers_local(
    dataset_name,
    vocab_size,
    src_lang,
    tgt_lang,
    src_column,
    tgt_column,
):
    """Train transformer tokenizers locally"""
    from datasets import load_dataset
    from tqdm import tqdm

    from tokenizer.bpe import BPE

    dataset_dir = dataset_name.replace("/", "_")
    tokenizer_dir = f"checkpoints/tokenizers/{dataset_dir}"
    src_tokenizer_path = f"{tokenizer_dir}/{src_lang}_bpe.pkl"
    tgt_tokenizer_path = f"{tokenizer_dir}/{tgt_lang}_bpe.pkl"

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    print(f"Training {src_lang} tokenizer...")
    src_texts = [
        ex[src_column] for ex in tqdm(dataset, desc=f"Loading {src_lang} texts")
    ]
    src_tokenizer = BPE(pattern=GPT2_PATTERN if src_lang == "en" else None)
    src_tokenizer.train(src_texts, vocab_size=vocab_size)

    print(f"Training {tgt_lang} tokenizer...")
    tgt_texts = [
        ex[tgt_column] for ex in tqdm(dataset, desc=f"Loading {tgt_lang} texts")
    ]
    tgt_tokenizer = BPE(pattern=None)
    tgt_tokenizer.train(tgt_texts, vocab_size=vocab_size)

    os.makedirs(tokenizer_dir, exist_ok=True)

    if os.path.exists(src_tokenizer_path):
        os.remove(src_tokenizer_path)
    if os.path.exists(tgt_tokenizer_path):
        os.remove(tgt_tokenizer_path)

    src_tokenizer.save(src_tokenizer_path)
    tgt_tokenizer.save(tgt_tokenizer_path)

    print(f"Tokenizers saved to {tokenizer_dir}")
    return tokenizer_dir


def train_gpt_tokenizer_local(
    dataset_name,
    dataset_config,
    vocab_size,
    text_column,
    num_samples,
):
    """Train GPT/Llama tokenizer locally"""
    from itertools import islice

    from datasets import load_dataset
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        pre_tokenizers,
        processors,
        trainers,
    )
    from tqdm import tqdm

    dataset_dir = dataset_name.replace("/", "_")
    tokenizer_dir = f"checkpoints/tokenizers/{dataset_dir}"
    tokenizer_path = f"{tokenizer_dir}/tokenizer.json"

    print(f"Loading dataset: {dataset_name} ({dataset_config})")
    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    def text_iterator():
        for example in tqdm(
            islice(dataset, num_samples), total=num_samples, desc="Loading texts"
        ):
            yield example[text_column]

    print(f"Training tokenizer with vocab_size={vocab_size}, num_samples={num_samples}")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    os.makedirs(tokenizer_dir, exist_ok=True)

    if os.path.exists(tokenizer_path):
        os.remove(tokenizer_path)

    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer_dir


@app.function(image=upload_image, volumes={"/vol": volume}, timeout=3600)
def upload_tokenizer_files(dataset_name: str, file_data: dict[str, bytes]):
    """Upload locally trained tokenizer files to Modal volume"""
    import shutil

    remote_dir = f"/vol/tokenizers/{dataset_name}"

    print(f"Uploading tokenizer to {remote_dir}")

    if os.path.exists(remote_dir):
        shutil.rmtree(remote_dir)

    os.makedirs(remote_dir, exist_ok=True)

    for filename, data in file_data.items():
        file_path = os.path.join(remote_dir, filename)
        with open(file_path, "wb") as f:
            f.write(data)
        print(f"  Uploaded: {filename}")

    volume.commit()

    print(f"Tokenizer uploaded successfully to {remote_dir}")
    return remote_dir


@app.local_entrypoint()
def main(
    model_type: str = "transformer",
    dataset: str = None,
    vocab_size: int = None,
    dataset_config: str = None,
):
    from pathlib import Path

    from config import Config

    if model_type == "transformer":
        config = Config.for_transformer()
    elif model_type == "gpt":
        config = Config.for_gpt()
    elif model_type == "llama":
        config = Config.for_llama()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    dataset_name = dataset or config.data.dataset_name
    vocab_size_value = vocab_size or config.data.vocab_size

    print(f"\n{'=' * 60}")
    print(f"Training tokenizer for {model_type.upper()}")
    print(f"{'=' * 60}\n")

    if model_type == "transformer":
        tokenizer_dir = train_transformer_tokenizers_local(
            dataset_name=dataset_name,
            vocab_size=vocab_size_value,
            src_lang=config.data.src_lang,
            tgt_lang=config.data.tgt_lang,
            src_column=config.data.src_column,
            tgt_column=config.data.tgt_column,
        )
    elif model_type == "gpt" or model_type == "llama":
        dataset_config_value = dataset_config or config.data.dataset_config
        tokenizer_dir = train_gpt_tokenizer_local(
            dataset_name=dataset_name,
            dataset_config=dataset_config_value,
            vocab_size=vocab_size_value,
            text_column=config.data.text_column,
            num_samples=config.data.tokenizer_train_samples,
        )

    print(f"\n{'=' * 60}")
    print("Uploading tokenizer to Modal volume")
    print(f"{'=' * 60}\n")

    # Read tokenizer files and prepare for upload
    tokenizer_path = Path(tokenizer_dir)
    dataset_name = tokenizer_path.name
    file_data = {}

    for file_path in tokenizer_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(tokenizer_path)
            with open(file_path, "rb") as f:
                file_data[str(relative_path)] = f.read()

    upload_tokenizer_files.remote(dataset_name, file_data)

    print(f"\n{'=' * 60}")
    print("Preparation complete!")
    print(f"{'=' * 60}\n")
