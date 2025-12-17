import modal
import torch

from model.transformer import Transformer
from scripts.config import Config
from utils.inference_pipeline import translate_batch

app = modal.App("llm-generation")

volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "datasets", "regex")
    .add_local_dir("model", remote_path="/root/llm-from-scratch/model")
    .add_local_dir("utils", remote_path="/root/llm-from-scratch/utils")
    .add_local_dir("block", remote_path="/root/llm-from-scratch/block")
    .add_local_dir("layer", remote_path="/root/llm-from-scratch/layer")
    .add_local_dir("component", remote_path="/root/llm-from-scratch/component")
    .add_local_dir("tokenizer", remote_path="/root/llm-from-scratch/tokenizer")
    .add_local_dir("scripts", remote_path="/root/llm-from-scratch/scripts")
)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/vol": volume},
    timeout=3600,
)
def generate(
    checkpoint_path: str, prompts: list[str], config: Config, strategy: str = "beam"
):
    from tokenizer.bpe import BPE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.model.model_type == "transformer":
        model = Transformer(config.model).to(device)

        tokenizer_dir = config.tokenizer_dir + "/bsd_en_ja"
        src_tokenizer = BPE.load(f"{tokenizer_dir}/en_bpe.pkl")
        tgt_tokenizer = BPE.load(f"{tokenizer_dir}/ja_bpe.pkl")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        outputs = translate_batch(
            model, prompts, src_tokenizer, tgt_tokenizer, config, strategy
        )

    elif config.model.model_type == "gpt":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")

    return outputs


@app.local_entrypoint()
def main(
    checkpoint_path: str,
    model_type: str = "transformer",
    prompt: str = None,
    prompts_file: str = None,
    strategy: str = "beam",
):
    if model_type == "transformer":
        config = Config.for_transformer()
    elif model_type == "gpt":
        config = Config.for_gpt()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if prompt:
        prompts = [prompt]
    elif prompts_file:
        with open(prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Either --prompt or --prompts-file must be provided")

    results = generate.remote(checkpoint_path, prompts, config, strategy)

    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\nInput {i + 1}: {prompt}")
        print(f"Output {i + 1}: {result}")
