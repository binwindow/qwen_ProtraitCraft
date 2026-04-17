"""
Dataset loading for Qwen-VL fine-tuning
Based on original: qwenvl/data/data_processor.py
Adapted for local JSON files (zijielou/PortraitCraft)
"""
import json
import os
import random
import re
import time
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from PIL import Image, ImageFile

# Allow loading very large images ( DecompressionBombError protection)
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Disable pixel limit check

from .rope2d import get_rope_index_3, get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"


def rank0_print(*args):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def update_processor_pixels(processor, data_args):
    """Update processor pixel settings (same as original implementation)."""
    ip = processor.image_processor

    min_pixels = data_args.get("min_pixels", 784)
    max_pixels = data_args.get("max_pixels", 50176)

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = min_pixels
        ip.max_pixels = max_pixels

    if hasattr(ip, "size"):
        ip.size.shortest_edge = min_pixels
        ip.size.longest_edge = max_pixels

    return processor


def _find_image_in_subdirs(base_path: str, image_name: str) -> str:
    """Find image in subdirectories (images_00, images_01, etc.)."""
    # If path exists directly, use it
    if os.path.exists(image_name):
        return image_name

    # Extract just the filename
    filename = os.path.basename(image_name)

    # Try direct path under base_path
    direct_path = os.path.join(base_path, filename)
    if os.path.exists(direct_path):
        return direct_path

    # Search in images_* subdirectories
    if os.path.exists(base_path):
        for subdir in os.listdir(base_path):
            subdir_path = os.path.join(base_path, subdir)
            if os.path.isdir(subdir_path) and subdir.startswith("images"):
                image_path = os.path.join(subdir_path, filename)
                if os.path.exists(image_path):
                    return image_path

    # Return original path if not found
    return image_name


def _build_conversation_from_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build conversation messages from HuggingFace dataset item format."""
    data_path = item.get("data_path", "")

    # Get image path
    image_path = item.get("image_path", "")
    if image_path:
        image_path = _find_image_in_subdirs(data_path, image_path)

    # Check if this is a conversation format or a criteria/question format
    if "conversations" in item:
        # Original conversation format
        return _build_conversation_from_conversations(item, image_path)
    elif "criteria" in item:
        # Training format with criteria scores
        return _build_criteria_conversation(item, image_path)
    elif "question" in item and "options" in item:
        # Test format with question and options
        return _build_qa_conversation(item, image_path)
    else:
        raise ValueError(f"Unknown item format: {list(item.keys())}")


def _build_conversation_from_conversations(item: Dict[str, Any], image_path: str) -> List[Dict[str, Any]]:
    """Build conversation from original conversations format."""
    image_pool = [{"type": "image", "image": image_path}] if image_path else []
    video_pool = []

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if image_pool:
                        content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if video_pool:
                        content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    return messages


def _build_criteria_conversation(item: Dict[str, Any], image_path: str) -> List[Dict[str, Any]]:
    """Build conversation from criteria format (training)."""
    criteria = item.get("criteria", {})

    # Build criteria text
    criteria_text = ""
    for name, details in criteria.items():
        if isinstance(details, dict):
            level = details.get("level", details.get("score", "N/A"))
            reason = details.get("reason", "")
            criteria_text += f"- {name}: {level}\n  Reason: {reason}\n"

    # Build user message
    user_content = [{"type": "image", "image": image_path}] if image_path else []
    user_content.append({
        "type": "text",
        "text": f"You are an aesthetics expert. Evaluate this image based on the following criteria:\n{criteria_text}\nProvide your assessment."
    })

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(item.get("criteria", {}), indent=2)}]}
    ]

    return messages


def _build_qa_conversation(item: Dict[str, Any], image_path: str) -> List[Dict[str, Any]]:
    """Build conversation from QA format (test)."""
    question = item.get("question", "")
    options = item.get("options", {})

    # Format options
    options_text = ""
    for key, value in options.items():
        options_text += f"{key}: {value}\n"

    # Build user message
    user_content = [{"type": "image", "image": image_path}] if image_path else []
    user_content.append({
        "type": "text",
        "text": f"{question}\n\nOptions:\n{options_text}\nProvide your answer."
    })

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": f"Answer: {item.get('answer', '')}"}]}
    ]

    return messages


def preprocess_qwen_visual(sources, processor) -> Dict:
    """Process sources with chat template."""
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    messages = _build_conversation_from_item(source)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start: ans_end + 2] = input_ids[0, ans_start: ans_end + 2]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning using HuggingFace datasets."""

    def __init__(self, processor, data_args: Dict):
        super().__init__()

        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args

        model_type = data_args.get("model_type", "qwen3vl")
        if model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {model_type} not supported")

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)

        dataset_path = data_args.get("dataset_path")
        split = data_args.get("dataset_split", "train")

        # Determine which JSON file to load based on split
        if split == "train":
            json_files = ["track_1_train.json"]
        else:
            json_files = ["track_1_test.json"]

        list_data_dict = []
        for json_file in json_files:
            json_path = os.path.join(dataset_path, json_file)
            if os.path.exists(json_path):
                rank0_print(f"Loading {json_path}")
                with open(json_path, "r") as f:
                    annotations = json.load(f)
                for ann in annotations:
                    ann["data_path"] = dataset_path
                list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        # Limit number of training samples if requested
        train_max_samples = data_args.get("train_max_samples")
        if train_max_samples is not None and train_max_samples < len(list_data_dict):
            list_data_dict = list_data_dict[:train_max_samples]
            rank0_print(f"Limited to {train_max_samples} samples for training")

        self.list_data_dict = list_data_dict

        self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        for attempt_idx in range(num_base_retries):
            try:
                item = self.list_data_dict[i]
                sample = self.item_fn([item])
                return sample
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        item = self.list_data_dict[i]
        sample = self.item_fn([item])
        return sample

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(sources, self.processor)

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=torch.cat(video_grid_thw, dim=0) if video_grid_thw else None,
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    return torch.cat(padded_tensors, dim=1)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        videos = [instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance]

        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            batch["image_grid_thw"] = torch.cat(grid_thw, dim=0) if grid_thw else None
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            batch["video_grid_thw"] = torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None

        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(itertools.chain(*[instance["attention_mask"] for instance in instances if "attention_mask" in instance]))
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        # Truncate to model_max_length (same as original implementation)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            batch["image_grid_thw"] = torch.cat(grid_thw, dim=0) if grid_thw else None
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        videos = [instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance]
        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            batch["video_grid_thw"] = torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None

        return batch


def make_supervised_data_module(processor, data_args: Dict) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)

    data_flatten = data_args.get("data_flatten", False)
    data_packing = data_args.get("data_packing", False)

    if data_flatten or data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
