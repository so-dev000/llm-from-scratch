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

### Option A: Modal (Cloud GPU)

#### 1. Prepare Tokenizer

```bash
modal run scripts/prepare.py --model-type=llama
```

#### 2. Train Model

```bash
modal run -d scripts/train.py --model-type=llama --run-name="llama-exp-1"
```

#### 3. Pull Model

```bash
modal run scripts/pull.py --run-name="llama-exp-1"
```

### Option B: Local Training (Free)

#### 1. Prepare Tokenizer

```bash
python -m scripts.prepare_local --model-type=llama --num-samples=100000
```

#### 2. Train Model

```bash
python -m scripts.train_local --model-type=llama --run-name="llama-exp-1"
```

### Option C: Google Colab (Free GPU)

1. Open `notebooks/train_llama_colab.ipynb` in Google Colab
2. Runtime > Change runtime type > GPU (T4)
3. Run all cells

### Common Steps

#### 3. Inference

```bash
python -m scripts.inference --run-name="llama-exp-1" --model-type=llama --mode=local
```

#### 4. Evaluation

```bash
python -m scripts.eval --run-name="llama-exp-1" --model-type=llama
```

#### 5. Visualization

```bash
python scripts/visualize_model.py llama
```

## Project Structure

```
scripts/
  config.py              # Centralized configuration
  prepare.py             # Tokenizer training (local) + upload to Modal
  train.py               # Model training (Modal)
  lightning_module.py    # PyTorch Lightning modules

model/
  transformer.py         # Transformer encoder-decoder
  gpt.py                 # GPT decoder-only
  llama.py               # Llama decoder-only with RoPE

block/
  llama_block.py         # Llama transformer block

component/
  attention.py           # Multi-head attention with GQA
  feed_forward_swiglu.py # SwiGLU feedforward
  rms_norm.py            # RMSNorm
  rotary_embedding.py    # Rotary Position Embedding

utils/
  training_pipeline.py   # Data loading and preprocessing
  decoding_strategy.py   # Inference strategies
```
