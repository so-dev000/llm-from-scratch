import os

import modal

app = modal.App("llm-data-preparation")

PROJECT_DIR = "/Users/nsota/llm-from-scratch"
LOCAL_DATA_DIR = f"{PROJECT_DIR}/data"

volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("regex", "tqdm")
    .add_local_dir(
        f"{PROJECT_DIR}/tokenizer", remote_path="/root/llm-from-scratch/tokenizer"
    )
)

DATASET_NAME = "ryo0634/bsd_ja_en"
VOCAB_SIZE = 8000


def prepare_data_locally():
    import sys

    sys.path.insert(0, PROJECT_DIR)

    from datasets import load_dataset

    from tokenizer.bpe import BPE

    tokenizer_dir = f"{LOCAL_DATA_DIR}/tokenizers/bsd_en_ja"
    en_tokenizer_path = f"{tokenizer_dir}/en_bpe.pkl"
    ja_tokenizer_path = f"{tokenizer_dir}/ja_bpe.pkl"

    if os.path.exists(en_tokenizer_path) and os.path.exists(ja_tokenizer_path):
        return tokenizer_dir

    dataset = load_dataset(DATASET_NAME, split="train")

    # GPT2-style pattern matching
    gpt2_pattern = (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?"
        r"[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )

    en_texts = [ex["en_sentence"] for ex in dataset]
    en_tokenizer = BPE(pattern=gpt2_pattern)
    en_tokenizer.train(en_texts, vocab_size=VOCAB_SIZE)

    ja_texts = [ex["ja_sentence"] for ex in dataset]
    ja_tokenizer = BPE(pattern=None)
    ja_tokenizer.train(ja_texts, vocab_size=VOCAB_SIZE)

    os.makedirs(tokenizer_dir, exist_ok=True)

    en_tokenizer.save(en_tokenizer_path)
    ja_tokenizer.save(ja_tokenizer_path)

    return tokenizer_dir


@app.function(image=image, volumes={"/vol": volume}, timeout=600)
def upload_files(local_files):
    for local_content, remote_path in local_files:
        remote_dir = os.path.dirname(remote_path)
        os.makedirs(remote_dir, exist_ok=True)
        with open(remote_path, "wb") as f:
            f.write(local_content)

    volume.commit()


@app.local_entrypoint()
def main():
    local_dir = prepare_data_locally()

    files_to_upload = []
    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        remote_path = f"/vol/tokenizers/bsd_en_ja/{filename}"
        with open(local_path, "rb") as f:
            files_to_upload.append((f.read(), remote_path))

    upload_files.remote(files_to_upload)
