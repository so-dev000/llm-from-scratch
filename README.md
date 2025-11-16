# LLM from Scratch

## Setup

```bash
cp .env.example .env
# Edit .env and set your HF_REPO_ID and HF_TOKEN
```

## Usage

### Train

```bash
# Train tokenizers
python -m scripts.train_tokenizers

# Train model
python -m scripts.train
```

### Translate

```bash
python -m scripts.translate
```

### Hugging Face Hub

```bash
# Upload checkpoints
python -m scripts.hub push

# Download checkpoints
python -m scripts.hub pull
```

## References

### Transformer

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Positional Encoding](https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6)

### Tokenizer

- [Normalization and pre-tokenization](https://huggingface.co/learn/llm-course/chapter6/4)
- [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Tiktokenizer](https://tiktokenizer.vercel.app/?model=gpt2)
