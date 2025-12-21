[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/so-dev000/llm-from-scratch)

# LLM from Scratch

## Blog

[Qiita @so_dev000](https://qiita.com/so_dev000)

## Wiki

[DeepWiki](https://deepwiki.com/so-dev000/llm-from-scratch)

## Setup

```bash
cp .env.example .env
# Edit .env and add: WANDB_API_KEY

# Install dependencies
uv sync

# Setup Modal
modal setup

# Create Modal secret
modal secret create wandb-secret WANDB_API_KEY=your-wandb-key
```

## Workflow

### Transformer (Translation Model)

#### 1. Prepare Data (Modal - First time only)

```bash
# Default: ryo0634/bsd_ja_en dataset
modal run -d scripts/prepare.py --model-type=transformer

# Custom dataset
modal run -d scripts/prepare.py --model-type=transformer --dataset=your/dataset --vocab-size=8000
```

#### 2. Train Model (Modal)

```bash
modal run -d scripts/train.py --model-type=transformer --run-name="exp-1"
```

#### 3. Pull Trained Model (Local)

```bash
modal run scripts/pull.py --run-name="exp-1"
```

#### 4. Evaluate Model (Local)

```bash
python -m scripts.eval --run-name="exp-1" --model-type=transformer
```

#### 5. Interactive Translation (Local)

```bash
python -m scripts.inference --run-name="exp-1" --model-type=transformer --mode=local
```

### GPT (Language Model)

#### 1. Prepare Data (Modal - First time only)

```bash
# Default: FineWeb-Edu dataset (sample-10BT, 1M samples for tokenizer)
modal run -d scripts/prepare.py --model-type=gpt

```

#### 2. Train Model (Modal)

```bash
modal run -d scripts/train.py --model-type=gpt --run-name="gpt-exp-1"
```

#### 3. Evaluate Model (Local)

```bash
# Not yet implemented - requires perplexity evaluation
python -m scripts.eval --run-name="gpt-exp-1" --model-type=gpt
```

#### 4. Interactive Generation (Local)

```bash
# Not yet implemented - requires generate_text()
python -m scripts.inference --run-name="gpt-exp-1" --model-type=gpt --mode=local
```

## Project Structure

```
scripts/
  config.py          # Unified config (Transformer & GPT)
  prepare.py         # Tokenizer preparation (Transformer & GPT)
  train.py           # Training (Transformer & GPT)
  eval.py            # Evaluation (BLEU/chrF++ for Transformer, Perplexity for GPT)
  inference.py       # Interactive/batch inference (Translation & Generation)
  lightning_module.py # PyTorch Lightning modules

model/
  transformer.py  # Encoder-Decoder Transformer
  gpt.py          # GPT (Decoder-only)

utils/
  training_pipeline.py  # DataModules (Transformer & GPT)
  inference_pipeline.py # Translation & Generation pipelines
  decoding_strategy.py  # Beam search, Greedy, Sampling
  collate.py            # Data collation
```

## References

### Transformer

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Positional Encoding](https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6)

### GPT

- [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)

- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)

- [THE CURIOUS CASE OF NEURAL TEXT DeGENERATION](https://arxiv.org/pdf/1904.09751)

- [Decoding Strategies in Large Language Models](https://huggingface.co/blog/mlabonne/decoding-strategies)

- [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)

### Tokenizer

- [Normalization and pre-tokenization](https://huggingface.co/learn/llm-course/chapter6/4)
- [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Tiktokenizer](https://tiktokenizer.vercel.app/?model=gpt2)

### Word2Vec

- [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)
