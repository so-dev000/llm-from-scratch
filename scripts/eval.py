import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from tqdm import tqdm

from model.gpt import GPT
from model.transformer import Transformer
from scripts.config import Config
from tokenizer.bpe import BPE
from utils.inference_pipeline import translate_batch

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


def evaluate_transformer(
    model, config, test_file, tokenizer_dir, output_path, strategy="beam"
):
    # device = next(model.parameters()).device

    src_tokenizer = BPE.load(f"{tokenizer_dir}/en_bpe.pkl")
    tgt_tokenizer = BPE.load(f"{tokenizer_dir}/ja_bpe.pkl")

    with open(test_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_data = [row for row in reader if row["en"].strip()]

    references = []
    hypotheses = []
    results = []

    batch_size = 32
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i : i + batch_size]
        en_sentences = [sample["en"] for sample in batch]

        translations = translate_batch(
            model, en_sentences, src_tokenizer, tgt_tokenizer, config, strategy=strategy
        )

        for sample, ja in zip(batch, translations):
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

    exact = sum(1 for r in results if r["ref"] == r["hyp"])
    exact_rate = exact / len(results) * 100

    print(
        f"\nchrF++: {score.score:.2f} | "
        f"Exact: {exact_rate:.1f}% ({exact}/{len(results)})"
    )

    groups = defaultdict(lambda: {"refs": [], "hyps": [], "exact": 0, "total": 0})

    for r in results:
        cat = r["category"]
        groups[cat]["refs"].append(r["ref"])
        groups[cat]["hyps"].append(r["hyp"])
        groups[cat]["total"] += 1
        if r["ref"] == r["hyp"]:
            groups[cat]["exact"] += 1

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

    with open(output_path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["en", "ref", "hyp", "category"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n-> {output_path}")


def evaluate_gpt(model, config, tokenizer_dir, test_data_loader):
    raise NotImplementedError("GPT evaluation not yet implemented")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="transformer")
    parser.add_argument("--checkpoint", type=str, default="best_model.ckpt")
    parser.add_argument("--test", type=str, default="data/test/test.csv")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--strategy", type=str, default="beam", choices=["beam", "greedy"]
    )
    args = parser.parse_args()

    run_name = args.run_name or find_latest_run()
    if not run_name:
        return

    if args.dataset is None:
        args.dataset = (
            "ryo0634/bsd_ja_en" if args.model_type == "transformer" else "openwebtext"
        )

    model, config, device = load_model_and_config(
        run_name, args.checkpoint, args.model_type, args.dataset
    )

    tokenizer_dir = get_tokenizer_dir(args.dataset)
    output_path = (
        args.output
        or Path(CHECKPOINT_BASE_DIR) / run_name / f"eval_results_{args.strategy}.csv"
    )

    if args.model_type == "transformer":
        evaluate_transformer(
            model, config, args.test, tokenizer_dir, output_path, args.strategy
        )
    elif args.model_type == "gpt":
        evaluate_gpt(model, config, tokenizer_dir, None)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


if __name__ == "__main__":
    main()
