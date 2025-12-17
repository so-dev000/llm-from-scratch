import argparse
import csv
from collections import Counter

from datasets import load_dataset

from scripts.config import Config


def categorize(text: str) -> str:
    words = len(text.split())
    if words <= 5:
        return "short"
    if words <= 15:
        return "medium"
    return "long"


def build_dataset(
    config: Config, output_path: str = "data/test/test.csv", seed: int = 42
) -> None:
    dataset = load_dataset(config.data.dataset_name, split="train")

    train_rest = dataset.train_test_split(test_size=0.15, seed=seed)
    val_test = train_rest["test"].train_test_split(test_size=0.33, seed=seed)
    test_data = val_test["test"]

    rows = []
    counts: Counter[str] = Counter()

    for item in test_data:
        en = item["en_sentence"].strip()
        ja = item["ja_sentence"].strip()
        if not en or not ja:
            continue

        category = categorize(en)
        rows.append({"en": en, "ja": ja, "category": category})
        counts[category] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["en", "ja", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Built {len(rows)} test samples -> {output_path}")
    for category in ("short", "medium", "long"):
        print(f"  {category:6s}: {counts.get(category, 0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/test/test.csv")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    config = Config.for_transformer()
    build_dataset(config, args.output, args.seed)
