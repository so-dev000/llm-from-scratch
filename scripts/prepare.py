import os
import pickle

import modal

app = modal.App("llm-data-preparation")

PROJECT_DIR = "/Users/nsota/llm-from-scratch"

data_volume = modal.Volume.from_name("llm-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets", "regex", "tqdm")
    .add_local_dir(
        f"{PROJECT_DIR}/tokenizer", remote_path="/root/llm-from-scratch/tokenizer"
    )
)

DATASET_NAME = "Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus"
VOCAB_SIZE = 8000
SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
}


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=1200,
    cpu=8,
    memory=16384,
)
def prepare_data():
    import sys

    sys.path.insert(0, "/root/llm-from-scratch")

    from datasets import load_dataset
    from tokenizer.bpe import BPE

    tokenizer_dir = "/data/tokenizers/jparacrawl"
    en_tokenizer_path = f"{tokenizer_dir}/en_bpe.pkl"
    ja_tokenizer_path = f"{tokenizer_dir}/ja_bpe.pkl"

    if os.path.exists(en_tokenizer_path) and os.path.exists(ja_tokenizer_path):
        return

    dataset = load_dataset(DATASET_NAME, split="train")

    en_texts = [item["english"] for item in dataset]
    ja_texts = [item["japanese"] for item in dataset]

    en_tokenizer = BPE(special_tokens=SPECIAL_TOKENS)
    en_tokenizer.train(en_texts, vocab_size=VOCAB_SIZE)

    ja_tokenizer = BPE(special_tokens=SPECIAL_TOKENS)
    ja_tokenizer.train(ja_texts, vocab_size=VOCAB_SIZE)

    os.makedirs(tokenizer_dir, exist_ok=True)

    with open(en_tokenizer_path, "wb") as f:
        pickle.dump(en_tokenizer, f)

    with open(ja_tokenizer_path, "wb") as f:
        pickle.dump(ja_tokenizer, f)

    data_volume.commit()


@app.local_entrypoint()
def main():
    prepare_data.remote()
