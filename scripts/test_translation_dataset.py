import pickle

from data.translation_dataset import TranslationDataset
from datasets import load_dataset

# load tokenizer
with open("checkpoints/tokenizers/bsd_ja_en/en_bpe.pkl", "rb") as f:
    en_tokenizer = pickle.load(f)

with open("checkpoints/tokenizers/bsd_ja_en/ja_bpe.pkl", "rb") as f:
    ja_tokenizer = pickle.load(f)

# load dataset
dataset = load_dataset("ryo0634/bsd_ja_en", cache_dir="./datasets")

train_dataset = TranslationDataset(
    data=dataset["train"],
    en_tokenizer=en_tokenizer,
    ja_tokenizer=ja_tokenizer,
    max_length=128,
)

# test
print(f"Dataset size: {len(train_dataset)}")
sample = train_dataset[0]
print(f"Source: {sample['src_text']}")
print(f"Target: {sample['tgt_text']}")
print(f"Source IDs: {sample['src']}")
print(f"Target IDs: {sample['tgt']}")
