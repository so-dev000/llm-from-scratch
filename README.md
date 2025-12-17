# LLM from Scratch

## Blog

[Qiita @so_dev000](https://qiita.com/so_dev000)

## Wiki

[DeepWiki](https://deepwiki.com/so-dev000/llm-from-scratch)

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
