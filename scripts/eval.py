import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from model.transformer import Transformer
from tokenizer.bpe import BPE
from utils.masking import combine_masks, create_causal_mask

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
# constants "Attention is All You Need" paper
BEAM_SIZE = 4
LENGTH_PENALTY = 0.6
MAX_OUTPUT_OFFSET = 50

TOKENIZER_DIR = "data/tokenizers/bsd_en_ja"
CHECKPOINT_BASE_DIR = "checkpoints/runs"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def length_penalty(length, alpha):
    return ((5 + length) / 6) ** alpha


def find_latest_run():
    runs_path = Path(CHECKPOINT_BASE_DIR)
    if not runs_path.exists():
        return None

    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    return max(run_dirs, key=lambda d: d.stat().st_mtime).name


def translate_sentence_beam(
    model,
    sentence,
    en_tokenizer,
    ja_tokenizer,
    beam_size=BEAM_SIZE,
    alpha=LENGTH_PENALTY,
):
    model.eval()

    src_ids = en_tokenizer.encode(sentence, add_special_tokens=True)
    max_length = len(src_ids) + MAX_OUTPUT_OFFSET

    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    src_padding_mask = src_tensor != PAD_IDX
    encoder_src_mask = src_padding_mask.unsqueeze(1) & src_padding_mask.unsqueeze(2)

    beams = [
        {
            "tokens": torch.tensor([[BOS_IDX]], device=device),
            "score": 0.0,
            "finished": False,
            "normalized_score": 0.0,
        }
    ]

    for step in range(max_length):
        all_candidates = []

        for beam in beams:
            # reached EOS
            if beam["finished"]:
                all_candidates.append(beam)
                continue

            # beam["tokens"]: (1, current_length)
            tgt_tensor = beam["tokens"]
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

            #  get last output
            # output: (1, current_length, vocab_size)
            logits = output[:, -1, :]  # (1, vocab_size)

            # TODO: implement by myself
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # get top-k tokens
            # top_k_log_probs: (1, beam_size)
            # top_k_indices: (1, beam_size)
            top_k_log_probs, top_k_indices = log_probs.topk(beam_size)

            # get new candidates from top-k tokens
            for i in range(beam_size):
                new_token = top_k_indices[0, i].item()
                new_log_prob = top_k_log_probs[0, i].item()

                new_tokens = torch.cat(
                    [tgt_tensor, torch.tensor([[new_token]], device=device)], dim=1
                )

                new_score = beam["score"] + new_log_prob

                is_finished = new_token == EOS_IDX

                current_length = new_tokens.size(1)
                penalty = length_penalty(current_length, alpha)
                normalized_score = new_score / penalty

                candidate = {
                    "tokens": new_tokens,
                    "score": new_score,
                    "finished": is_finished,
                    "normalized_score": normalized_score,
                }

                all_candidates.append(candidate)

        sorted_candidates = sorted(
            all_candidates,
            key=lambda x: x["normalized_score"],
            reverse=True,
        )
        # get next beam (top-k)
        beams = sorted_candidates[:beam_size]

        if all(beam["finished"] for beam in beams):
            break

    best_beam = max(beams, key=lambda b: b["normalized_score"])

    tgt_ids = best_beam["tokens"].squeeze(0).tolist()

    # remove special tokens
    filtered_ids = [id for id in tgt_ids if id not in [PAD_IDX, BOS_IDX, EOS_IDX]]
    return ja_tokenizer.decode(filtered_ids).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--test", type=str, default="data/test/test.csv")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(TOKENIZER_DIR):
        print(f"Tokenizers not found at {TOKENIZER_DIR}")
        return

    en_tokenizer_path = f"{TOKENIZER_DIR}/en_bpe.pkl"
    ja_tokenizer_path = f"{TOKENIZER_DIR}/ja_bpe.pkl"

    if not os.path.exists(en_tokenizer_path) or not os.path.exists(ja_tokenizer_path):
        print(f"Tokenizer files not found: {en_tokenizer_path}, {ja_tokenizer_path}")
        return

    en_tokenizer = BPE.load(en_tokenizer_path)
    ja_tokenizer = BPE.load(ja_tokenizer_path)

    run_name = args.run_name or find_latest_run()
    if not run_name:
        print("No training runs found")
        return

    checkpoint_path = Path(CHECKPOINT_BASE_DIR) / run_name / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    src_vocab_size = checkpoint["src_vocab_size"]
    tgt_vocab_size = checkpoint["tgt_vocab_size"]
    model_dim = checkpoint["model_dim"]
    encoder_layers = checkpoint["encoder_layers"]
    decoder_layers = checkpoint["decoder_layers"]

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        model_dim=model_dim,
        encoder_num=encoder_layers,
        decoder_num=decoder_layers,
        padding_idx=PAD_IDX,
    ).to(device)

    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {
            key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)

    # Load test data
    with open(args.test, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_data = [row for row in reader if row["en"].strip()]

    # Evaluate
    references = []
    hypotheses = []
    results = []

    for sample in tqdm(test_data):
        ja = translate_sentence_beam(model, sample["en"], en_tokenizer, ja_tokenizer)

        references.append([sample["ja"]])
        hypotheses.append(ja)

        results.append(
            {
                "en": sample["en"],
                "ref": sample["ja"],
                "hyp": ja,
                "category": sample["category"],
            }
        )

    # Compute BLEU
    bleu = BLEU(tokenize="char")
    bleu_score = bleu.corpus_score(hypotheses, references)

    # Exact match
    exact = sum(1 for r in results if r["ref"] == r["hyp"])
    exact_rate = exact / len(results) * 100

    # Overall
    print(
        f"\nBLEU: {bleu_score.score:.2f} | "
        f"Exact: {exact_rate:.1f}% ({exact}/{len(results)})"
    )

    # Group by category
    groups = defaultdict(lambda: {"refs": [], "hyps": [], "exact": 0, "total": 0})

    for r in results:
        cat = r["category"]
        groups[cat]["refs"].append([r["ref"]])
        groups[cat]["hyps"].append(r["hyp"])
        groups[cat]["total"] += 1
        if r["ref"] == r["hyp"]:
            groups[cat]["exact"] += 1

    # By category
    for cat in ["short", "medium", "long"]:
        if cat in groups:
            g = groups[cat]
            score = bleu.corpus_score(g["hyps"], g["refs"])
            exact_pct = g["exact"] / g["total"] * 100
            print(
                f"  {cat:6s}: BLEU {score.score:5.1f} | "
                f"Exact {exact_pct:4.1f}% ({g['total']})"
            )

    # Save
    output_path = args.output
    if not output_path:
        output_path = Path(CHECKPOINT_BASE_DIR) / run_name / "eval_results.csv"

    with open(output_path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["en", "ref", "hyp", "category"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n-> {output_path}")


if __name__ == "__main__":
    main()
