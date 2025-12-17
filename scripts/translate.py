import argparse
from pathlib import Path

import torch

from model.transformer import Transformer
from scripts.config import Config
from tokenizer.bpe import BPE
from utils.inference_pipeline import translate_sentence

TOKENIZER_DIR = "data/tokenizers/bsd_en_ja"
CHECKPOINT_BASE_DIR = "checkpoints/runs"


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
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    en_tokenizer_path = f"{TOKENIZER_DIR}/en_bpe.pkl"
    ja_tokenizer_path = f"{TOKENIZER_DIR}/ja_bpe.pkl"

    en_tokenizer = BPE.load(en_tokenizer_path)
    ja_tokenizer = BPE.load(ja_tokenizer_path)

    run_name = args.run_name or find_latest_run()
    if not run_name:
        print("No training runs found")
        return

    checkpoint_path = Path(CHECKPOINT_BASE_DIR) / run_name / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = Config.for_transformer()
    config.model.src_vocab_size = checkpoint["src_vocab_size"]
    config.model.tgt_vocab_size = checkpoint["tgt_vocab_size"]
    config.model.model_dim = checkpoint["model_dim"]
    config.model.encoder_layers = checkpoint["encoder_layers"]
    config.model.decoder_layers = checkpoint["decoder_layers"]

    model = Transformer(config.model).to(device)

    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {
            key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)

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
