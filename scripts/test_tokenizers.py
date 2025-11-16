import pickle

# load trained tokenizer
with open("checkpoints/tokenizers/bsd_ja_en/en_bpe.pkl", "rb") as f:
    en_tokenizer = pickle.load(f)

with open("checkpoints/tokenizers/bsd_ja_en/ja_bpe.pkl", "rb") as f:
    ja_tokenizer = pickle.load(f)

# test
en_text = "Hello, how are you?"
ja_text = "こんにちは、元気ですか？"

en_ids = en_tokenizer.encode(en_text, add_special_tokens=True)
ja_ids = ja_tokenizer.encode(ja_text, add_special_tokens=True)

print(f"English: {en_text}")
print(f"  Tokens: {en_ids}")
print(f"  Decoded: {en_tokenizer.decode(en_ids, skip_special_tokens=True)}")

print(f"\nJapanese: {ja_text}")
print(f"  Tokens: {ja_ids}")
print(f"  Decoded: {ja_tokenizer.decode(ja_ids, skip_special_tokens=True)}")
