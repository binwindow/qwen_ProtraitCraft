#!/usr/bin/env python3
"""
Convert test_results.json to submission format
Poor -> A, Medium -> B, Good -> C
"""
import json
import os

level_map = {"Poor": "A", "Medium": "B", "Good": "C"}


def convert_criteria(criteria):
    """Convert criteria levels to submission format."""
    if not criteria:
        return {}

    converted = {}
    for k, v in criteria.items():
        if isinstance(v, dict) and "level" in v:
            old_level = v["level"]
            new_level = level_map.get(old_level, "NOT_RES")
            converted[k] = {"level": new_level}
        else:
            converted[k] = {"level": "NOT_RES"}
    return converted


def convert_file(input_path, output_path=None):
    """Convert single file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    for item in data:
        new_item = {
            "image_path": item["image_path"],
            "criteria": convert_criteria(item.get("criteria", {})),
            "total_score": item.get("total_score"),
            "question": item.get("question"),
            "options": item.get("options"),
            "answer": item.get("answer")
        }
        converted.append(new_item)

    if output_path is None:
        output_path = input_path.replace(".json", "_submission.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} items")
    print(f"Output: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/pretrain/test/test_results.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    convert_file(input_file, output_file)
