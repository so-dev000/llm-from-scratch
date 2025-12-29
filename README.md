# LLM from Scratch

Implementation of Transformer, GPT, and Llama models from scratch using PyTorch.

## Resources

- Blog: [Qiita @so_dev000](https://qiita.com/so_dev000)
- Wiki: [DeepWiki](https://deepwiki.com/so-dev000/llm-from-scratch)

## Setup

```bash
cp .env.example .env
# Edit .env and add: WANDB_API_KEY

uv sync

modal setup
modal secret create wandb-secret WANDB_API_KEY=your-wandb-key
```

## Workflow

### 1. Prepare Data

```bash
modal run -d scripts/prepare.py --model-type=llama
```

### 2. Train Model

```bash
modal run -d scripts/train.py --model-type=llama --run-name="llama-exp-1"
```

### 3. Pull Trained Model

```bash
modal run scripts/pull.py --run-name="llama-exp-1"
```

### 4. Evaluate Model

```bash
python -m scripts.eval --run-name="llama-exp-1" --model-type=llama
```

### 5. Interactive Inference

```bash
python -m scripts.inference --run-name="llama-exp-1" --model-type=llama --mode=local
```

## Project Structure

```
scripts/
  config.py
  train.py
  lightning_module.py

model/
  transformer.py
  gpt.py
  llama.py

component/
  attention.py
  feed_forward_swiglu.py
  rms_norm.py
  rotary_embedding.py
```
