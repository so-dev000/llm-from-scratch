# LLM from Scratch

## Setup

```bash
cp .env.example .env
# Edit .env and add: WANDB_API_KEY

# Install Modal
pip install modal
modal setup

# Create Modal secret
modal secret create wandb-secret WANDB_API_KEY=your-wandb-key
```

## Workflow

### Prepare Data (Modal - First time only)

```bash
modal run -d scripts/prepare.py
```

### Train Model (Modal)

```bash
# Background training
modal run -d scripts/train.py --run-name="experiment-1"

# Check status
modal app list
modal app logs llm-training
```

### Pull Trained Model (Local)

```bash
# List available runs
modal run scripts/pull.py --list-only=true

# Pull specific run
modal run scripts/pull.py --run-name="experiment-1"

# Pull tokenizers only
modal run scripts/pull.py
```

### Translate (Local)

```bash
# Use latest run
python -m scripts.translate

# Use specific run
python -m scripts.translate --run-name="experiment-1"

# Use specific checkpoint
python -m scripts.translate --run-name="experiment-1" --checkpoint="checkpoint_epoch_10.pt"
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
