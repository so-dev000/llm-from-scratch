import argparse
import csv
import random
from collections import defaultdict

from datasets import load_dataset


def categorize(text):
    words = len(text.split())
    if words <= 5:
        return "short"
    elif words <= 15:
        return "medium"
    else:
        return "long"


def build_dataset(output_path="data/test/test.csv", target_size=500, seed=42):
    random.seed(seed)

    dataset = load_dataset("ryo0634/bsd_ja_en", split="train")

    # 3-way split: train 85%, val 10%, test 5%
    train_rest = dataset.train_test_split(test_size=0.15, seed=42)
    val_test = train_rest["test"].train_test_split(test_size=0.33, seed=42)
    test_data = val_test["test"]

    candidates = defaultdict(list)

    for item in test_data:
        en = item["en_sentence"].strip()
        ja = item["ja_sentence"].strip()
        if en and ja:
            category = categorize(en)
            candidates[category].append({"en": en, "ja": ja})

    distribution = {
        "short": 0.20,
        "medium": 0.40,
        "long": 0.40,
    }

    samples = []

    for category, ratio in distribution.items():
        count = int(target_size * ratio)
        available = candidates[category]

        if len(available) < count:
            count = len(available)

        sampled = random.sample(available, count)

        for item in sampled:
            samples.append(
                {
                    "en": item["en"],
                    "ja": item["ja"],
                    "category": category,
                }
            )

    random.shuffle(samples)

    with open(output_path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["en", "ja", "category"])
        writer.writeheader()
        writer.writerows(samples)

    print(f"Built {len(samples)} test samples -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/test/test.csv")
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    build_dataset(args.output, args.size, args.seed)
