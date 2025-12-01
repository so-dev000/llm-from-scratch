import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from tqdm import tqdm

from model.transformer import Transformer
from tokenizer.bpe import BPE
from utils.inference import PAD_IDX, get_device, translate_sentence_beam

TOKENIZER_DIR = "data/tokenizers/bsd_en_ja"
CHECKPOINT_BASE_DIR = "checkpoints/runs"

device = get_device()


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

        references.append(sample["ja"])
        hypotheses.append(ja)

        results.append(
            {
                "en": sample["en"],
                "ref": sample["ja"],
                "hyp": ja,
                "category": sample["category"],
            }
        )

    metric = CHRF(word_order=2)
    score = metric.corpus_score(hypotheses, [references])

    # Exact match
    exact = sum(1 for r in results if r["ref"] == r["hyp"])
    exact_rate = exact / len(results) * 100

    # Overall
    print(
        f"\nchrF++: {score.score:.2f} | "
        f"Exact: {exact_rate:.1f}% ({exact}/{len(results)})"
    )

    # Group by category
    groups = defaultdict(lambda: {"refs": [], "hyps": [], "exact": 0, "total": 0})

    for r in results:
        cat = r["category"]
        groups[cat]["refs"].append(r["ref"])
        groups[cat]["hyps"].append(r["hyp"])
        groups[cat]["total"] += 1
        if r["ref"] == r["hyp"]:
            groups[cat]["exact"] += 1

    # By category
    for cat in ["short", "medium", "long"]:
        if cat in groups:
            g = groups[cat]
            cat_metric = CHRF(word_order=2)
            cat_score = cat_metric.corpus_score(g["hyps"], [g["refs"]])
            exact_pct = g["exact"] / g["total"] * 100
            print(
                f"  {cat:6s}: chrF++ {cat_score.score:5.1f} | "
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
