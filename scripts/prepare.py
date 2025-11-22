import os

import modal

app = modal.App("llm-data-preparation")

PROJECT_DIR = "/Users/nsota/llm-from-scratch"
LOCAL_DATA_DIR = f"{PROJECT_DIR}/data"

volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11")

DATASET_NAME = "Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus"
VOCAB_SIZE = 8000
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def batch_iterator(dataset, lang, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][lang]


def prepare_data_locally():
    from datasets import load_dataset
    from tokenizers import Regex, Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Split
    from tokenizers.trainers import BpeTrainer

    tokenizer_dir = f"{LOCAL_DATA_DIR}/tokenizers/jparacrawl"
    en_tokenizer_path = f"{tokenizer_dir}/en_bpe.json"
    ja_tokenizer_path = f"{tokenizer_dir}/ja_bpe.json"

    if os.path.exists(en_tokenizer_path) and os.path.exists(ja_tokenizer_path):
        return tokenizer_dir

    dataset = load_dataset(DATASET_NAME, split="train")

    # GPT2-style pattern matching
    gpt2_pattern = (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?"
        r"[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )

    en_tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    en_tokenizer.pre_tokenizer = Split(
        Regex(gpt2_pattern), behavior="isolated", invert=True
    )
    en_trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    en_tokenizer.train_from_iterator(
        batch_iterator(dataset, "english"), trainer=en_trainer
    )

    ja_tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    # DO NOT SPLIT JAPANESE
    ja_trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    ja_tokenizer.train_from_iterator(
        batch_iterator(dataset, "japanese"), trainer=ja_trainer
    )

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
    prepare_data_locally()
    local_dir = f"{LOCAL_DATA_DIR}/tokenizers/jparacrawl"

    files_to_upload = []
    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        remote_path = f"/vol/tokenizers/jparacrawl/{filename}"
        with open(local_path, "rb") as f:
            files_to_upload.append((f.read(), remote_path))

    upload_files.remote(files_to_upload)
