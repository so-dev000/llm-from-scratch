import os

import modal

app = modal.App("llm-data-preparation")

volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("regex", "tqdm", "datasets")
    .add_local_dir("tokenizer", remote_path="/root/llm-from-scratch/tokenizer")
)

GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?"
    r"[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def prepare_transformer_tokenizers(
    dataset_name="ryo0634/bsd_ja_en",
    vocab_size=8000,
    src_lang="en",
    tgt_lang="ja",
    src_column="en_sentence",
    tgt_column="ja_sentence",
):
    from datasets import load_dataset

    from tokenizer.bpe import BPE

    dataset_dir = dataset_name.replace("/", "_")
    tokenizer_dir = f"data/tokenizers/{dataset_dir}"

    src_tokenizer_path = f"{tokenizer_dir}/{src_lang}_bpe.pkl"
    tgt_tokenizer_path = f"{tokenizer_dir}/{tgt_lang}_bpe.pkl"

    if os.path.exists(src_tokenizer_path) and os.path.exists(tgt_tokenizer_path):
        return tokenizer_dir

    dataset = load_dataset(dataset_name, split="train")

    src_texts = [ex[src_column] for ex in dataset]
    src_tokenizer = BPE(pattern=GPT2_PATTERN if src_lang == "en" else None)
    src_tokenizer.train(src_texts, vocab_size=vocab_size)

    tgt_texts = [ex[tgt_column] for ex in dataset]
    tgt_tokenizer = BPE(pattern=None)
    tgt_tokenizer.train(tgt_texts, vocab_size=vocab_size)

    os.makedirs(tokenizer_dir, exist_ok=True)
    src_tokenizer.save(src_tokenizer_path)
    tgt_tokenizer.save(tgt_tokenizer_path)

    return tokenizer_dir


def prepare_gpt_tokenizer(
    dataset_name="openwebtext",
    vocab_size=50257,
    text_column="text",
    sample_size=100000,
):
    raise NotImplementedError("GPT tokenizer preparation not yet implemented")


@app.function(image=image, volumes={"/vol": volume}, timeout=600)
def upload_files(local_files):
    for local_content, remote_path in local_files:
        remote_dir = os.path.dirname(remote_path)
        os.makedirs(remote_dir, exist_ok=True)
        with open(remote_path, "wb") as f:
            f.write(local_content)

    volume.commit()


@app.local_entrypoint()
def main(
    model_type: str = "transformer",
    dataset: str = None,
    vocab_size: int = None,
):
    if model_type == "transformer":
        dataset_name = dataset or "ryo0634/bsd_ja_en"
        vs = vocab_size or 8000
        local_dir = prepare_transformer_tokenizers(
            dataset_name=dataset_name,
            vocab_size=vs,
            src_lang="en",
            tgt_lang="ja",
            src_column="en_sentence",
            tgt_column="ja_sentence",
        )
    elif model_type == "gpt":
        dataset_name = dataset or "openwebtext"
        vs = vocab_size or 50257
        local_dir = prepare_gpt_tokenizer(
            dataset_name=dataset_name,
            vocab_size=vs,
            text_column="text",
            sample_size=100000,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    dataset_dir = dataset_name.replace("/", "_")
    files_to_upload = []
    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        remote_path = f"/vol/tokenizers/{dataset_dir}/{filename}"
        with open(local_path, "rb") as f:
            files_to_upload.append((f.read(), remote_path))

    upload_files.remote(files_to_upload)
