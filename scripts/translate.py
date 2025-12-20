import argparse
from pathlib import Path

import torch

from model.transformer import Transformer
from scripts.config import Config
from tokenizer.bpe import BPE
from utils.inference_pipeline import translate_sentence

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best_model.ckpt")
    parser.add_argument("--dataset", type=str, default="ryo0634/bsd_ja_en")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_dir = get_tokenizer_dir(args.dataset)
    en_tokenizer_path = f"{tokenizer_dir}/en_bpe.pkl"
    ja_tokenizer_path = f"{tokenizer_dir}/ja_bpe.pkl"

    en_tokenizer = BPE.load(en_tokenizer_path)
    ja_tokenizer = BPE.load(ja_tokenizer_path)

    run_name = args.run_name or find_latest_run()
    if not run_name:
        return

    checkpoint_path = Path(CHECKPOINT_BASE_DIR) / run_name / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    hyper_params = checkpoint.get("hyper_parameters", {})
    config = Config.for_transformer()
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

    while True:
        try:
            en = input("EN: ").strip()
            if en.lower() in ["exit", "quit", "q"]:
                break
            if not en:
                continue

            ja = translate_sentence(
                model, en, en_tokenizer, ja_tokenizer, config, strategy="beam"
            )
            print(f"JA: {ja}\n")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
