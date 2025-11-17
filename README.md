# LLM from Scratch

## Setup

```bash
cp .env.example .env
# Edit .env:
# - HF_REPO_ID: your-username/repo-name
# - HF_TOKEN: get from https://huggingface.co/settings/tokens (with write permission)
# - WANDB_API_KEY: get from https://wandb.ai/settings#api
```

### Modal Setup

```bash
# Setup Modal secrets
modal secret create huggingface-secret HF_REPO_ID=your-username/repo-name HF_TOKEN=your-hf-token
modal secret create wandb-secret WANDB_API_KEY=your-wandb-key
```

## Usage

### Train

```bash
# 1. Train tokenizers
python -m scripts.train_tokenizers

# 2. Upload tokenizers to Hugging Face Hub
python -m scripts.hub push

# 3a. Train model locally
python -m scripts.train

# 3b. Train on Modal
# (downloads tokenizers, uploads trained model automatically)

# run training in background (can close terminal/sleep)
modal run -d scripts/train_on_modal.py::train

# Check running task
modal app list
```

### Translate

```bash
python -m scripts.translate
```

### Hugging Face Hub

```bash
# Upload entire checkpoints folder
python -m scripts.hub push

# Download entire checkpoints folder
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
