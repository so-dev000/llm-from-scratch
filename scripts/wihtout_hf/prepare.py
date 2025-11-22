import os
import pickle
import sys

import modal

app = modal.App("llm-data-preparation")

PROJECT_DIR = "/Users/nsota/llm-from-scratch"
LOCAL_DATA_DIR = f"{PROJECT_DIR}/data"

volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11")

DATASET_NAME = "ryo0634/bsd_ja_en"
VOCAB_SIZE = 8000
SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
}


def prepare_data_locally():
    sys.path.insert(0, PROJECT_DIR)

    from datasets import load_dataset

    from tokenizer.bpe import BPE

    tokenizer_dir = f"{LOCAL_DATA_DIR}/tokenizers/bsd_en_ja"
    en_tokenizer_path = f"{tokenizer_dir}/en_bpe.pkl"
    ja_tokenizer_path = f"{tokenizer_dir}/ja_bpe.pkl"

    if os.path.exists(en_tokenizer_path) and os.path.exists(ja_tokenizer_path):
        return tokenizer_dir

    dataset = load_dataset(DATASET_NAME, split="train")

    en_texts = [item["en_sentence"] for item in dataset]
    ja_texts = [item["ja_sentence"] for item in dataset]

    en_tokenizer = BPE(special_tokens=SPECIAL_TOKENS)
    en_tokenizer.train(en_texts, vocab_size=VOCAB_SIZE)

    ja_tokenizer = BPE(special_tokens=SPECIAL_TOKENS)
    ja_tokenizer.train(ja_texts, vocab_size=VOCAB_SIZE)

    os.makedirs(tokenizer_dir, exist_ok=True)

    with open(en_tokenizer_path, "wb") as f:
        pickle.dump(en_tokenizer, f)

    with open(ja_tokenizer_path, "wb") as f:
        pickle.dump(ja_tokenizer, f)

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
    local_dir = f"{LOCAL_DATA_DIR}/tokenizers/bsd_en_ja"

    files_to_upload = []
    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        remote_path = f"/vol/tokenizers/bsd_en_ja/{filename}"
        with open(local_path, "rb") as f:
            files_to_upload.append((f.read(), remote_path))

    upload_files.remote(files_to_upload)
