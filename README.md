# LLM from Scratch

## Blog

[Qiita @so_dev000](https://qiita.com/so_dev000)

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

### 1. Prepare Data (Modal - First time only)

```bash
modal run -d scripts/prepare.py
```

### 2. Train Model (Modal)

```bash
modal run -d scripts/train.py --run-name="experiment-1"
```

### 3. Pull Trained Model (Local)

```bash
modal run scripts/pull.py --run-name="experiment-1"
```

### 4. Evaluate Model (Local)

```bash
python -m scripts.eval --run-name="experiment-1"
```

### 5. Translate (Local)

```bash
python -m scripts.translate --run-name="experiment-1"
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
- [Decoding Strategies in Large Language Models](https://huggingface.co/blog/mlabonne/decoding-strategies)
