import argparse
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.transformer import Transformer
from utils.masking import combine_masks, create_causal_mask

MAX_LENGTH = 64
MODEL_DIM = 512
ENCODER_LAYERS = 6
DECODER_LAYERS = 6
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

TOKENIZER_DIR = "checkpoints/tokenizers/jparacrawl"
CHECKPOINT_BASE_DIR = "checkpoints/runs"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def find_latest_run():
    runs_path = Path(CHECKPOINT_BASE_DIR)
    if not runs_path.exists():
        return None

    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    return max(run_dirs, key=lambda d: d.stat().st_mtime).name


def translate_sentence(model, sentence, en_tokenizer, ja_tokenizer):
    model.eval()

    src_encoding = en_tokenizer.encode(sentence)
    src_ids = src_encoding.ids
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    src_padding_mask = src_tensor != PAD_IDX
    encoder_src_mask = src_padding_mask.unsqueeze(1) & src_padding_mask.unsqueeze(2)

    tgt_tensor = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)

    for _ in range(MAX_LENGTH):
        tgt_len = tgt_tensor.size(1)
        causal_mask = create_causal_mask(tgt_len, device=device)
        tgt_padding_mask = tgt_tensor != PAD_IDX
        tgt_combined_mask = combine_masks(tgt_padding_mask, causal_mask)

        with torch.no_grad():
            output = model(
                src_tensor,
                tgt_tensor,
                encoder_src_mask=encoder_src_mask,
                decoder_src_mask=src_padding_mask,
                tgt_mask=tgt_combined_mask,
            )

        pred_token_id = output.argmax(dim=-1)[:, -1].item()

        tgt_tensor = torch.cat(
            [
                tgt_tensor,
                torch.tensor([[pred_token_id]], dtype=torch.long, device=device),
            ],
            dim=1,
        )

        if pred_token_id == EOS_IDX:
            break

    tgt_ids = tgt_tensor.squeeze(0).tolist()
    filtered_ids = [id for id in tgt_ids if id not in [PAD_IDX, BOS_IDX, EOS_IDX]]
    return ja_tokenizer.decode(filtered_ids).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    args = parser.parse_args()

    if not os.path.exists(TOKENIZER_DIR):
        print(f"Tokenizers not found at {TOKENIZER_DIR}")
        return

    en_tokenizer_path = f"{TOKENIZER_DIR}/en_bpe.json"
    ja_tokenizer_path = f"{TOKENIZER_DIR}/ja_bpe.json"

    if not os.path.exists(en_tokenizer_path) or not os.path.exists(ja_tokenizer_path):
        print(f"Tokenizer files not found: {en_tokenizer_path}, {ja_tokenizer_path}")
        return

    en_tokenizer = Tokenizer.from_file(en_tokenizer_path)
    ja_tokenizer = Tokenizer.from_file(ja_tokenizer_path)

    run_name = args.run_name or find_latest_run()
    if not run_name:
        print("No training runs found")
        return

    checkpoint_path = Path(CHECKPOINT_BASE_DIR) / run_name / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    vocab_size = max(en_tokenizer.get_vocab_size(), ja_tokenizer.get_vocab_size())

    model = Transformer(
        vocab_size=vocab_size,
        model_dim=MODEL_DIM,
        encoder_num=ENCODER_LAYERS,
        decoder_num=DECODER_LAYERS,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

            ja = translate_sentence(model, en, en_tokenizer, ja_tokenizer)
            print(f"JA: {ja}\n")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
