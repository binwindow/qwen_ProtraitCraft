#!/usr/bin/env python3
"""
Qwen-VL Test Script
Based on original: evaluation/evaluation_multi.py
Supports testing pretrained model or fine-tuned model from checkpoint
Supports resume from中断 and computes metrics after all samples done
"""
import argparse
import json
import os
import sys
from PIL import Image

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed
from src.evaluation.metrics import evaluate_and_save

Image.MAX_IMAGE_PIXELS = None


# Detailed criteria descriptions for enhanced prompt
CRITERIA_DESCRIPTIONS = {
    "Color Harmony": {
        "poor": "Clashing or chaotic colors; oversaturated; no hierarchy.",
        "medium": "Mostly harmonious but dull or slightly crowded.",
        "good": "Balanced, vivid, complementary colors with strong contrast."
    },
    "Visual Style Consistency": {
        "poor": "Mixed styles; inconsistent lighting and tone.",
        "medium": "Mostly consistent with minor mismatches.",
        "good": "Cohesive style across lighting, tone, and details."
    },
    "Sharpness": {
        "poor": "Blurry; misfocused; key details lost.",
        "medium": "Subject clear but edges soft or slightly noisy.",
        "good": "Crisp focus; fine details clearly visible."
    },
    "Light and Shadow Modeling": {
        "poor": "Harsh or flat lighting; no depth.",
        "medium": "Pleasant but lacks depth or smooth transitions.",
        "good": "Natural, directional light with strong depth and form."
    },
    "Creativity and Originality": {
        "poor": "Cliché; generic; no novelty.",
        "medium": "Standard idea with decent execution.",
        "good": "Unique concept or perspective; visually fresh."
    },
    "Exposure Control": {
        "poor": "Over/underexposed; detail loss.",
        "medium": "Mostly correct; minor highlight/shadow issues.",
        "good": "Well-balanced exposure; full detail retained."
    },
    "Application of Classical Composition Principles": {
        "poor": "Random framing; awkward crop; unstable balance.",
        "medium": "Basic composition; lacks refinement.",
        "good": "Well-structured; strong framing and guidance."
    },
    "Depth of Field and Layering": {
        "poor": "Flat; cluttered background; no separation.",
        "medium": "Some separation; limited depth.",
        "good": "Clear subject separation; strong depth and layers."
    },
    "Visual Center Stability": {
        "poor": "Distracting elements; unclear focus.",
        "medium": "Subject stands out but slight distractions.",
        "good": "Clear focal point; stable and attention-grabbing."
    },
    "Visual Flow Guidance": {
        "poor": "Chaotic flow; unclear viewing path.",
        "medium": "Mostly smooth but occasionally interrupted.",
        "good": "Natural, smooth guidance toward the subject."
    },
    "Structural Support Stability": {
        "poor": "Loose structure; visually unstable.",
        "medium": "Basic alignment; average stability.",
        "good": "Strong geometric support; stable composition."
    },
    "Appropriateness of Negative Space": {
        "poor": "Overcrowded; no breathing space.",
        "medium": "Limited space; barely sufficient.",
        "good": "Well-balanced space; comfortable and intentional."
    },
    "Subject Integrity": {
        "poor": "Awkward cuts; incomplete subject.",
        "medium": "Tight crop but acceptable.",
        "good": "Complete, clean subject presentation."
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL Test")

    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path. If provided, load fine-tuned model; "
                             "otherwise use pretrained model directly.")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name. Auto-detected from ckpt path if not provided.")

    parser.add_argument("--model_name_or_path", type=str, default="./source/Qwen3-VL-4B-Instruct")

    parser.add_argument("--input_json", type=str, default=None,
                        help="Path to test JSON file (default based on dataset_type)")
    parser.add_argument("--images_path", type=str,
                        default="./source/PortraitCraft_dataset",
                        help="Base path for images in JSON")
    parser.add_argument("--output_json", type=str, default="test_results.json",
                        help="Output JSON file path")

    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_image_size", type=int, default=2048)
    parser.add_argument("--prompt_type", type=str, default="simple",
                        choices=["simple", "enhanced"],
                        help="Prompt type: simple (level=x) or enhanced (detailed descriptions)")
    parser.add_argument("--dataset_type", type=str, default="test",
                        choices=["test", "val"],
                        help="Dataset type: test (with question/options) or val (criteria only)")
    parser.add_argument("--metrics_max_samples", type=int, default=None,
                        help="Limit samples for metrics computation (None = all)")

    args = parser.parse_args()
    return args


def resize_keep_aspect(image_path, max_size=2048):
    """Resize image keeping aspect ratio."""
    img = Image.open(image_path)
    w, h = img.size

    if max(w, h) <= max_size:
        return img

    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    return img


def _build_enhanced_criteria_text():
    """Build enhanced criteria text with detailed descriptions."""
    lines = []
    for name, levels in CRITERIA_DESCRIPTIONS.items():
        lines.append(f"{name}:")
        lines.append(f"  Poor: {levels['poor']}")
        lines.append(f"  Medium: {levels['medium']}")
        lines.append(f"  Good: {levels['good']}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(item, prompt_type="simple", dataset_type="test"):
    """Build evaluation prompt from item."""
    criteria = item.get("criteria", {})
    question = item.get("question", "")
    options = item.get("options", {})

    if prompt_type == "enhanced":
        criteria_text = _build_enhanced_criteria_text()
    elif dataset_type == "val":
        # Val dataset: use placeholder x to avoid leaking ground truth labels
        criteria_text = "\n".join([
            f"{k}: level=x"
            for k in criteria
        ])
    else:
        criteria_text = "\n".join([
            f"{k}: level={v['level']}"
            for k, v in criteria.items()
        ])

    if dataset_type == "val":
        # Validation dataset: no question/options/answer
        prompt = f"""
You are an expert visual aesthetics evaluator.

You MUST analyze the image and output STRICT JSON ONLY.

---

TASKS:
1. Predict overall aesthetic score (1-100) based on visual evidence
2. Predict each criterion level: Good / Medium / Poor

---

IMAGE CRITERIA:
{criteria_text}

---

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "total_score": "<integer 1-100 inferred from image>",
  "criteria": {{
    "Color Harmony": "<Good|Medium|Poor>",
    "Visual Style Consistency": "<Good|Medium|Poor>",
    "Sharpness": "<Good|Medium|Poor>",
    "Light and Shadow Modeling": "<Good|Medium|Poor>",
    "Creativity and Originality": "<Good|Medium|Poor>",
    "Exposure Control": "<Good|Medium|Poor>",
    "Application of Classical Composition Principles": "<Good|Medium|Poor>",
    "Depth of Field and Layering": "<Good|Medium|Poor>",
    "Visual Center Stability": "<Good|Medium|Poor>",
    "Visual Flow Guidance": "<Good|Medium|Poor>",
    "Structural Support Stability": "<Good|Medium|Poor>",
    "Appropriateness of Negative Space": "<Good|Medium|Poor>",
    "Subject Integrity": "<Good|Medium|Poor>"
  }}
}}

---

IMPORTANT RULES:

- DO NOT copy any fixed value pattern.
- DO NOT output identical labels for all criteria.
- Each criterion MUST be judged independently from image evidence.
- total_score MUST NOT be a template value; it must reflect real visual quality.
- If unsure, choose Medium instead of guessing Good/Poor blindly.

---

Return ONLY valid JSON. No explanation. No markdown.
"""
    else:
        # Test dataset: includes question/options/answer
        options_text = "\n".join([
            f"{k}. {v}"
            for k, v in options.items()
        ])

        prompt = f"""
You are an expert visual aesthetics evaluator.

You MUST analyze the image and output STRICT JSON ONLY.

---

TASKS:
1. Predict overall aesthetic score (1-100) based on visual evidence
2. Predict each criterion level: Good / Medium / Poor
3. Answer the multiple-choice question (A/B/C/D)

---

IMAGE CRITERIA:
{criteria_text}

---

QUESTION:
{question}

---

OPTIONS:
{options_text}

---

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "total_score": "<integer 1-100 inferred from image>",
  "criteria": {{
    "Color Harmony": "<Good|Medium|Poor>",
    "Visual Style Consistency": "<Good|Medium|Poor>",
    "Sharpness": "<Good|Medium|Poor>",
    "Light and Shadow Modeling": "<Good|Medium|Poor>",
    "Creativity and Originality": "<Good|Medium|Poor>",
    "Exposure Control": "<Good|Medium|Poor>",
    "Application of Classical Composition Principles": "<Good|Medium|Poor>",
    "Depth of Field and Layering": "<Good|Medium|Poor>",
    "Visual Center Stability": "<Good|Medium|Poor>",
    "Visual Flow Guidance": "<Good|Medium|Poor>",
    "Structural Support Stability": "<Good|Medium|Poor>",
    "Appropriateness of Negative Space": "<Good|Medium|Poor>",
    "Subject Integrity": "<Good|Medium|Poor>"
  }},
  "answer": "<A|B|C|D>"
}}

---

IMPORTANT RULES:

- DO NOT copy any fixed value pattern.
- DO NOT output identical labels for all criteria.
- Each criterion MUST be judged independently from image evidence.
- total_score MUST NOT be a template value; it must reflect real visual quality.
- answer MUST be grounded in visible evidence.
- If unsure, choose Medium instead of guessing Good/Poor blindly.

---

Return ONLY valid JSON. No explanation. No markdown.
"""
    return prompt.strip()


def extract_json(text):
    """Extract JSON from model output."""
    try:
        text = text.replace("```json", "").replace("```", "")
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(text[start:end + 1])
    except:
        return None


def convert_criteria_to_submission_format(criteria):
    """
    Convert criteria from model output format to submission format.
    Poor -> A, Medium -> B, Good -> C
    """
    if not criteria:
        return {}

    level_map = {"Poor": "A", "Medium": "B", "Good": "C"}
    converted = {}

    for k, v in criteria.items():
        if isinstance(v, str):
            level = level_map.get(v, "NOT_RES")
        elif isinstance(v, dict) and "level" in v:
            level = level_map.get(v["level"], "NOT_RES")
        else:
            level = "NOT_RES"
        converted[k] = {"level": level}

    return converted


def load_done_set(output_json):
    """Load already processed samples from output file."""
    done_set = set()

    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                for x in data:
                    done_set.add(x["image_path"])
        except:
            pass

    print(f"Loaded {len(done_set)} already processed samples")
    return done_set


def save_results(results, output_json):
    """Save results atomically."""
    tmp_path = output_json + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_json)


class DemoServer:
    """Model inference server."""

    def __init__(self, model_path, device, max_new_tokens=512, temperature=0.2, top_p=0.9,
                 lora_enable=False, base_model_path=None):
        print(f"Loading model on {device}...")

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        if lora_enable and base_model_path:
            # Load base model and merge LoRA adapter
            from peft import PeftModel
            print(f"Loading LoRA adapter from: {model_path}")
            print(f"Loading base model from: {base_model_path}")

            base_model = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model = self.model.merge_and_unload()
            self.model.to(device)
            self.processor = AutoProcessor.from_pretrained(base_model_path)
            print(f"LoRA model merged and loaded successfully")
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
            ).to(device)
            self.processor = AutoProcessor.from_pretrained(model_path)

        print(f"Model loaded successfully")

    @torch.no_grad()
    def infer_one(self, image_path, prompt):
        """Run inference on single image."""
        img = resize_keep_aspect(image_path, max_size=2048)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p
        )

        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        if "assistant" in output:
            output = output.split("assistant")[-1].strip()

        return output


def find_image(base_path, image_name):
    """Find image in subdirectories."""
    direct_path = os.path.join(base_path, image_name)
    if os.path.exists(direct_path):
        return direct_path

    images_dir = os.path.dirname(base_path.rstrip('/'))
    if os.path.isdir(images_dir):
        for subdir in os.listdir(images_dir):
            subdir_path = os.path.join(images_dir, subdir)
            if os.path.isdir(subdir_path) and subdir.startswith('images'):
                image_path = os.path.join(subdir_path, image_name)
                if os.path.exists(image_path):
                    return image_path

    # Also check subdirs of the base_path itself
    if os.path.isdir(base_path):
        for subdir in os.listdir(base_path):
            subdir_path = os.path.join(base_path, subdir)
            if os.path.isdir(subdir_path) and subdir.startswith('images'):
                image_path = os.path.join(subdir_path, image_name)
                if os.path.exists(image_path):
                    return image_path

    return None


def main():
    args = parse_args()

    # Set input_json based on dataset_type if not provided
    if args.input_json is None:
        if args.dataset_type == "val":
            args.input_json = "./source/PortraitCraft_dataset/track_1_val.json"
        else:
            args.input_json = "./source/PortraitCraft_dataset/track_1_test.json"

    set_seed(args.seed)

    if args.ckpt:
        ckpt_dir = os.path.dirname(args.ckpt)
        exp_name = os.path.basename(os.path.dirname(ckpt_dir))
        model_path = args.ckpt
        print(f"Auto-detected exp_name: {exp_name} from ckpt path")
        print(f"Loading fine-tuned model from: {model_path}")

        # Load experiment config to check if LoRA was used
        config_path = os.path.join(args.save_dir, exp_name, "log", "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                exp_config = json.load(f)
            lora_enable = exp_config.get("lora_enable", False)
            base_model_path = exp_config.get("model_name_or_path", args.model_name_or_path)
            print(f"Experiment config loaded: lora_enable={lora_enable}")
        else:
            lora_enable = False
            base_model_path = args.model_name_or_path
            print(f"Config not found at {config_path}, assuming non-LoRA")
    else:
        exp_name = args.exp_name or "pretrained_test"
        model_path = args.model_name_or_path
        lora_enable = False
        base_model_path = args.model_name_or_path
        print(f"Testing pretrained model: {model_path}")

    test_dir = os.path.join(args.save_dir, exp_name, "test")
    os.makedirs(test_dir, exist_ok=True)

    # Determine output filename based on dataset_type and ckpt info
    if args.output_json == "test_results.json":
        # Get checkpoint name for inclusion in output file
        ckpt_name = ""
        if args.ckpt:
            ckpt_name = os.path.basename(args.ckpt)  # e.g., checkpoint-6500-srcc0.3371

        if args.dataset_type == "val":
            base_name = "val_results"
        else:
            base_name = "test_results"

        if ckpt_name:
            output_json = os.path.join(test_dir, f"{base_name}_{ckpt_name}.json")
        else:
            output_json = os.path.join(test_dir, f"{base_name}.json")
    else:
        output_json = args.output_json
        if not os.path.isabs(output_json):
            output_json = os.path.join(test_dir, output_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading test data from: {args.input_json}")
    with open(args.input_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Total test samples: {len(test_data)}")

    done_set = load_done_set(output_json)

    remaining_data = [item for item in test_data if item["image_path"] not in done_set]
    print(f"Remaining samples to process: {len(remaining_data)}")

    if len(remaining_data) == 0:
        print("All samples already processed!")
    else:
        server = DemoServer(
            model_path=model_path,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            lora_enable=lora_enable,
            base_model_path=base_model_path,
        )

        results = []
        if os.path.exists(output_json):
            try:
                with open(output_json, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except:
                results = []

        from tqdm import tqdm
        for item in tqdm(remaining_data, desc="Testing"):
            image_path = find_image(args.images_path, item["image_path"])

            if not image_path:
                tqdm.write(f"Image not found: {item['image_path']}")
                continue

            prompt = build_prompt(item, prompt_type=args.prompt_type, dataset_type=args.dataset_type)

            try:
                raw = server.infer_one(image_path, prompt)
                parsed = extract_json(raw)

                if parsed:
                    total_score = parsed.get("total_score")
                    if isinstance(total_score, str):
                        total_score = int(total_score)
                    record = {
                        "image_path": item["image_path"],
                        "total_score": total_score,
                        "criteria": convert_criteria_to_submission_format(parsed.get("criteria")),
                    }
                    if args.dataset_type == "test":
                        record["question"] = item.get("question")
                        record["options"] = item.get("options")
                        record["answer"] = parsed.get("answer")
                    results.append(record)
                    save_results(results, output_json)
                else:
                    print(f"Failed to parse JSON for: {item['image_path']}")

            except Exception as e:
                print(f"Error processing {item['image_path']}: {e}")
                continue

        print(f"\nInference completed!")
        print(f"Total processed: {len(results)}/{len(test_data)}")

    # Compute metrics only for val dataset (test dataset has no ground truth for comparison)
    if args.dataset_type == "val":
        print("\nComputing metrics...")

        # Build metrics filename with same ckpt info as results file
        ckpt_name = os.path.basename(args.ckpt) if args.ckpt else ""
        if ckpt_name:
            metrics_file = os.path.join(os.path.dirname(output_json), f"val_metrics_{ckpt_name}.json")
        else:
            metrics_file = os.path.join(os.path.dirname(output_json), "val_metrics.json")

        metrics = evaluate_and_save(
            input_json=args.input_json,
            pred_json=output_json,
            metrics_path=metrics_file,
            max_samples=args.metrics_max_samples,
        )

        print("\n=== Test Metrics ===")
        print(f"SRCC (Spearman):    {metrics['srcc']:.4f}")
        print(f"PLCC (Pearson):     {metrics['plcc']:.4f}")
        print(f"Level Acc:          {metrics['level_acc']:.4f}")
        print(f"QA Acc:             {metrics['qa_acc']:.4f}")
        print(f"Samples:            {metrics['num_samples']}/{metrics['num_total']}")
        print(f"Metrics saved to:   {metrics_file}")
    else:
        print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()