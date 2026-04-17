#!/usr/bin/env python3
"""
Create validation dataset from train dataset.
- Extract 2000 samples from track_1_train.json
- Convert score to level: 0<=s<5->A, 5<=s<7->B, 7<=s<=10->C
- No question/options/answer (train doesn't have them)
"""
import json
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed


def score_to_level(score):
    """Convert score (0-10) to level (A/B/C)."""
    if score < 5:
        return "A"
    elif score < 7:
        return "B"
    else:
        return "C"


def extract_filename(full_path):
    """Extract filename from full path."""
    return os.path.basename(full_path)


def create_val_dataset(
    train_json="./source/PortraitCraft_dataset/track_1_train.json",
    output_json="./source/PortraitCraft_dataset/track_1_val.json",
    val_size=2000,
    seed=42
):
    """Create validation dataset."""
    set_seed(seed)

    print(f"Loading train data from: {train_json}")
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Total train samples: {len(train_data)}")

    # Random sample
    val_data = random.sample(train_data, min(val_size, len(train_data)))
    print(f"Sampled {len(val_data)} validation samples")

    # Convert to val format
    converted = []
    for item in val_data:
        # Extract filename from full path
        image_path = extract_filename(item["image_path"])

        # Convert criteria scores to levels
        criteria = {}
        for name, info in item["criteria"].items():
            score = info.get("score", 0)
            level = score_to_level(score)
            criteria[name] = {"level": level}

        # Convert total_score to same scale if needed (train has 0-100, test has 0-100)
        # Keep as is, but the test evaluation uses rank correlation so scale doesn't matter much
        total_score = item.get("total_score", 0)

        converted_item = {
            "image_path": image_path,
            "criteria": criteria,
            "total_score": total_score,
        }
        converted.append(converted_item)

    # Save
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Saved validation dataset to: {output_json}")
    print(f"Validation samples: {len(converted)}")

    # Verify format
    sample = converted[0]
    print(f"\nSample validation item:")
    print(f"  image_path: {sample['image_path']}")
    print(f"  total_score: {sample['total_score']}")
    print(f"  criteria keys: {list(sample['criteria'].keys())[:3]}...")
    print(f"  sample criteria: Color Harmony -> {sample['criteria'].get('Color Harmony')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str,
                        default="./source/PortraitCraft_dataset/track_1_train.json")
    parser.add_argument("--output_json", type=str,
                        default="./source/PortraitCraft_dataset/track_1_val.json")
    parser.add_argument("--val_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    create_val_dataset(
        train_json=args.train_json,
        output_json=args.output_json,
        val_size=args.val_size,
        seed=args.seed
    )
