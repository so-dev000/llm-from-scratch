import os

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


def main(
    model_type: str = "transformer",
    dataset: str = None,
    vocab_size: int = None,
    dataset_config: str = None,
):
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
        train_transformer_tokenizers_local(
            dataset_name=dataset_name,
            vocab_size=vocab_size_value,
            src_lang=config.data.src_lang,
            tgt_lang=config.data.tgt_lang,
            src_column=config.data.src_column,
            tgt_column=config.data.tgt_column,
        )
    elif model_type == "gpt" or model_type == "llama":
        dataset_config_value = dataset_config or config.data.dataset_config
        train_gpt_tokenizer_local(
            dataset_name=dataset_name,
            dataset_config=dataset_config_value,
            vocab_size=vocab_size_value,
            text_column=config.data.text_column,
            num_samples=config.data.tokenizer_train_samples,
        )

    print(f"\n{'=' * 60}")
    print("Preparation complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="transformer")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--dataset-config", type=str, default=None)

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        dataset=args.dataset,
        vocab_size=args.vocab_size,
        dataset_config=args.dataset_config,
    )
