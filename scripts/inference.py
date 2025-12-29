import argparse
from pathlib import Path

import modal
import torch
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from tokenizers import Tokenizer

from model.gpt import GPT
from model.llama import Llama
from model.transformer import Transformer
from scripts.config import Config
from tokenizer.bpe import BPE
from utils.inference_pipeline import (
    generate_batch,
    generate_text,
    translate_batch,
    translate_sentence,
)

app = modal.App("llm-inference")

volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "datasets", "regex")
    .add_local_dir("model", remote_path="/root/model")
    .add_local_dir("utils", remote_path="/root/utils")
    .add_local_dir("block", remote_path="/root/block")
    .add_local_dir("layer", remote_path="/root/layer")
    .add_local_dir("component", remote_path="/root/component")
    .add_local_dir("tokenizer", remote_path="/root/tokenizer")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

CHECKPOINT_BASE_DIR = "checkpoints/runs"


def get_tokenizer_dir(dataset_name):
    dataset_dir = dataset_name.replace("/", "_")
    return f"checkpoints/tokenizers/{dataset_dir}"


def find_latest_run():
    runs_path = Path(CHECKPOINT_BASE_DIR)
    if not runs_path.exists():
        return None

    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    return max(run_dirs, key=lambda d: d.stat().st_mtime).name


def load_model_and_config(run_name, checkpoint_name, model_type, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(CHECKPOINT_BASE_DIR) / run_name / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hyper_params = checkpoint.get("hyper_parameters", {})

    if model_type == "transformer":
        config = Config.for_transformer()
        config.data.dataset_name = dataset
        config.model.src_vocab_size = hyper_params.get("model.src_vocab_size", 8000)
        config.model.tgt_vocab_size = hyper_params.get("model.tgt_vocab_size", 8000)
        model = Transformer(config.model).to(device)
    elif model_type == "gpt":
        config = Config.for_gpt()
        config.data.dataset_name = dataset
        config.model.vocab_size = hyper_params.get("model.vocab_size", 50257)
        model = GPT(config.model).to(device)
    elif model_type == "llama":
        config = Config.for_llama()
        config.data.dataset_name = dataset
        config.model.vocab_size = hyper_params.get("model.vocab_size", 32000)
        config.model.model_dim = hyper_params.get(
            "model.model_dim", config.model.model_dim
        )
        config.model.num_layers = hyper_params.get(
            "model.num_layers", config.model.num_layers
        )
        config.model.num_heads = hyper_params.get(
            "model.num_heads", config.model.num_heads
        )
        config.model.num_kv_heads = hyper_params.get(
            "model.num_kv_heads", config.model.num_kv_heads
        )
        config.model.feedforward_dim = hyper_params.get(
            "model.feedforward_dim", config.model.feedforward_dim
        )
        config.model.max_seq_len = hyper_params.get(
            "model.max_seq_len", config.model.max_seq_len
        )
        config.model.dropout = hyper_params.get("model.dropout", config.model.dropout)
        config.model.norm_eps = hyper_params.get(
            "model.norm_eps", config.model.norm_eps
        )
        config.model.rope_theta = hyper_params.get(
            "model.rope_theta", config.model.rope_theta
        )
        model = Llama(config.model).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state_dict[key[6:]] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    return model, config, device


@app.function(image=image, gpu="L40S", volumes={"/vol": volume}, timeout=3600)
def run_inference_remote(run_name, prompts, model_type, dataset, checkpoint, strategy):
    model, config, device = load_model_and_config(
        run_name, checkpoint, model_type, dataset
    )

    tokenizer_dir = get_tokenizer_dir(dataset)

    if model_type == "transformer":
        src_tokenizer = BPE.load(f"{tokenizer_dir}/en_bpe.pkl")
        tgt_tokenizer = BPE.load(f"{tokenizer_dir}/ja_bpe.pkl")

        results = translate_batch(
            model, prompts, src_tokenizer, tgt_tokenizer, config, strategy
        )
    elif model_type == "gpt":
        tokenizer = Tokenizer.from_file(f"{tokenizer_dir}/tokenizer.json")

        results = generate_batch(model, prompts, tokenizer, config, strategy)
    elif model_type == "llama":
        tokenizer = Tokenizer.from_file(f"{tokenizer_dir}/tokenizer.json")

        results = generate_batch(model, prompts, tokenizer, config, strategy)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return results


def run_inference_local(
    run_name, model_type, dataset, checkpoint, interactive, strategy="sampling"
):
    model, config, device = load_model_and_config(
        run_name, checkpoint, model_type, dataset
    )

    tokenizer_dir = get_tokenizer_dir(dataset)

    if model_type == "transformer":
        src_tokenizer = BPE.load(f"{tokenizer_dir}/en_bpe.pkl")
        tgt_tokenizer = BPE.load(f"{tokenizer_dir}/ja_bpe.pkl")

        if interactive:
            while True:
                try:
                    text = input("EN: ").strip()
                    if text.lower() in ["exit", "quit", "q"]:
                        break
                    if not text:
                        continue

                    result = translate_sentence(
                        model, text, src_tokenizer, tgt_tokenizer, config
                    )
                    print(f"JA: {result}\n")
                except KeyboardInterrupt:
                    break
    elif model_type == "gpt":
        tokenizer = Tokenizer.from_file(f"{tokenizer_dir}/tokenizer.json")

        if interactive:
            print("GPT Interactive Mode (type 'exit', 'quit', or 'q' to exit)")
            print("=" * 80)

            history = InMemoryHistory()
            style = Style.from_dict(
                {
                    "prompt": "#00aa00 bold",
                }
            )

            while True:
                try:
                    text = prompt(
                        "Prompt> ", history=history, style=style, multiline=False
                    ).strip()

                    if text.lower() in ["exit", "quit", "q"]:
                        print("Exiting...")
                        break
                    if not text:
                        continue

                    print("\nGenerating...", flush=True)
                    result = generate_text(
                        model, text, tokenizer, config, strategy=strategy
                    )
                    print(f"\nGenerated:\n{result}")
                    print("\n" + "=" * 80, flush=True)
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
                except EOFError:
                    print("\n\nExiting...")
                    break
    elif model_type == "llama":
        tokenizer = Tokenizer.from_file(f"{tokenizer_dir}/tokenizer.json")

        if interactive:
            print("Llama Interactive Mode (type 'exit', 'quit', or 'q' to exit)")
            print("=" * 80)

            history = InMemoryHistory()
            style = Style.from_dict(
                {
                    "prompt": "#00aa00 bold",
                }
            )

            while True:
                try:
                    text = prompt(
                        "Prompt> ", history=history, style=style, multiline=False
                    ).strip()

                    if text.lower() in ["exit", "quit", "q"]:
                        print("Exiting...")
                        break
                    if not text:
                        continue

                    print("\nGenerating...", flush=True)
                    result = generate_text(
                        model, text, tokenizer, config, strategy=strategy
                    )
                    print(f"\nGenerated:\n{result}")
                    print("\n" + "=" * 80, flush=True)
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
                except EOFError:
                    print("\n\nExiting...")
                    break


@app.local_entrypoint()
def main(
    run_name: str = None,
    model_type: str = "transformer",
    dataset: str = None,
    checkpoint: str = "best_model.ckpt",
    prompt: str = None,
    prompts_file: str = None,
    strategy: str = "sampling",
    mode: str = "remote",
):
    run_name = run_name or find_latest_run()
    if not run_name:
        return

    if dataset is None:
        if model_type == "transformer":
            config = Config.for_transformer()
        elif model_type == "gpt":
            config = Config.for_gpt()
        elif model_type == "llama":
            config = Config.for_llama()
        else:
            config = Config.for_gpt()
        dataset = config.data.dataset_name

    if mode == "remote":
        if prompt:
            prompts = [prompt]
        elif prompts_file:
            with open(prompts_file) as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("--prompt or --prompts-file required for remote mode")

        results = run_inference_remote.remote(
            run_name, prompts, model_type, dataset, checkpoint, strategy
        )

        for p, result in zip(prompts, results):
            print(f"Input: {p}")
            print(f"Output: {result}\n")
    else:
        run_inference_local(run_name, model_type, dataset, checkpoint, interactive=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="transformer")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best_model.ckpt")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--strategy", type=str, default="sampling")
    parser.add_argument(
        "--mode", type=str, default="local", choices=["local", "remote"]
    )
    args = parser.parse_args()

    if args.mode == "local":
        dataset = args.dataset
        if dataset is None:
            if args.model_type == "transformer":
                config = Config.for_transformer()
            elif args.model_type == "gpt":
                config = Config.for_gpt()
            elif args.model_type == "llama":
                config = Config.for_llama()
            else:
                config = Config.for_gpt()
            dataset = config.data.dataset_name

        run_inference_local(
            args.run_name or find_latest_run(),
            args.model_type,
            dataset,
            args.checkpoint,
            interactive=True,
            strategy=args.strategy,
        )
    else:
        main(
            args.run_name,
            args.model_type,
            args.dataset,
            args.checkpoint,
            args.prompt,
            args.prompts_file,
            args.strategy,
            args.mode,
        )
