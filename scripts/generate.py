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
    .add_local_dir("model", remote_path="/root/model")
    .add_local_dir("utils", remote_path="/root/utils")
    .add_local_dir("block", remote_path="/root/block")
    .add_local_dir("layer", remote_path="/root/layer")
    .add_local_dir("component", remote_path="/root/component")
    .add_local_dir("tokenizer", remote_path="/root/tokenizer")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/vol": volume},
    timeout=3600,
)
def generate(run_name: str, prompts: list[str], config: Config, strategy: str = "beam"):
    from tokenizer.bpe import BPE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.model.model_type == "transformer":
        dataset_dir = config.data.dataset_name.replace("/", "_")
        tokenizer_dir = f"{config.tokenizer_dir}/{dataset_dir}"
        src_tokenizer = BPE.load(f"{tokenizer_dir}/en_bpe.pkl")
        tgt_tokenizer = BPE.load(f"{tokenizer_dir}/ja_bpe.pkl")

        checkpoint_path = f"{config.checkpoint_dir}/{run_name}/best_model.ckpt"
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        hyper_params = checkpoint.get("hyper_parameters", {})
        config.model.src_vocab_size = hyper_params.get("model.src_vocab_size", 8000)
        config.model.tgt_vocab_size = hyper_params.get("model.tgt_vocab_size", 8000)

        model = Transformer(config.model).to(device)

        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]
                new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)

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
    run_name: str,
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

    results = generate.remote(run_name, prompts, config, strategy)

    for i, (p, result) in enumerate(zip(prompts, results)):
        print(f"\nInput {i + 1}: {p}")
        print(f"Output {i + 1}: {result}")
