import os
import pickle

from datasets import load_dataset
from tokenizer.bpe import BPE

# load dataset
dataset = load_dataset("ryo0634/bsd_ja_en", cache_dir="./datasets")
print("loaded dataset")

en_texts = []
ja_texts = []

for data in dataset["train"]:
    en_texts.append(data["en_sentence"])
    ja_texts.append(data["ja_sentence"])

print(f"Loaded {len(en_texts)} sentence pairs")

# train en tokenizer
print("start training en_tokenizer")
en_tokenizer = BPE()
en_tokenizer.train(en_texts, vocab_size=6000)
print(f"English vocab size: {len(en_tokenizer.vocab)}")

# train ja tokenizer
print("start training ja_tokenizer")
ja_tokenizer = BPE()
ja_tokenizer.train(ja_texts, vocab_size=6000)
print(f"Japanese vocab size: {len(ja_tokenizer.vocab)}")

os.makedirs("checkpoints/tokenizers", exist_ok=True)
with open("checkpoints/tokenizers/bsd_ja_en/en_bpe.pkl", "wb") as f:
    pickle.dump(en_tokenizer, f)

with open("checkpoints/tokenizers/bsd_ja_en/ja_bpe.pkl", "wb") as f:
    pickle.dump(ja_tokenizer, f)

print("✓ Tokenizers saved to checkpoints/tokenizers/")
