# LLM from Scratch

## Setup

```bash
uv sync
modal setup
modal secret create wandb-secret WANDB_API_KEY=your-key
```

## Training Pipeline

### 1. Prepare Tokenizer

```bash
modal run scripts/prepare.py --model-type=llama
```

### 2. Preprocess Data (Local)

```bash
python scripts/preprocess.py
```

### 3. Upload Preprocessed Data to Modal

```bash
modal volume put llm-from-scratch data/preprocessed/HuggingFaceFW_fineweb-edu /preprocessed/HuggingFaceFW_fineweb-edu
```

### 4. Train Model

```bash
# GPU: H200 (140GB), batch_size=320
modal run -d scripts/train.py --model-type=llama --run-name="llama-50M-2.5M-data"
```

### 5. Pull Model

```bash
# Pull best model only
modal run scripts/pull.py --run-name llama-50M-2.5M-data

# Pull specific checkpoint file
modal run scripts/pull.py --run-name llama-50M-2.5M-data --filename epoch-0.ckpt --no-best-model
```

## Inference

```bash
# Interactive mode
python -m scripts.inference --run-name llama-50M-2.5M-data --model-type llama

# With epoch checkpoint
python -m scripts.inference --run-name llama-50M-2.5M-data --model-type llama --checkpoint epoch-0.ckpt

# With specific prompt
python -m scripts.inference --run-name llama-50M-2.5M-data --model-type llama --prompt "Hello, how are you?"
```
