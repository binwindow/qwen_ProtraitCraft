"""
Validation Evaluator for PortraitCraft
Similar to test.py evaluation logic, but for training validation
"""
import json
import os
from PIL import Image

import torch
from tqdm import tqdm

from .metrics import compute_correlation_metrics


class ValidationEvaluator:
    """Validation evaluator with model inference."""

    def __init__(
        self,
        model,
        processor,
        device,
        val_json_path: str,
        images_path: str,
        max_samples: int = 200,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        save_dir: str = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.val_json_path = val_json_path
        self.images_path = images_path
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.save_dir = save_dir

        self.model.eval()

    def resize_keep_aspect(self, image_path, max_size=2048):
        """Resize image keeping aspect ratio."""
        img = Image.open(image_path)
        w, h = img.size
        if max(w, h) <= max_size:
            return img
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.BILINEAR)

    def find_image(self, image_name):
        """Find image in subdirectories."""
        direct_path = os.path.join(self.images_path, image_name)
        if os.path.exists(direct_path):
            return direct_path

        if os.path.isdir(self.images_path):
            for subdir in os.listdir(self.images_path):
                subdir_path = os.path.join(self.images_path, subdir)
                if os.path.isdir(subdir_path) and subdir.startswith('images'):
                    image_path = os.path.join(subdir_path, image_name)
                    if os.path.exists(image_path):
                        return image_path
        return None

    def build_prompt(self, item):
        """Build evaluation prompt."""
        criteria = item.get("criteria", {})

        criteria_text = "\n".join([
            f"{k}: level={v['level']}"
            for k, v in criteria.items()
        ])

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
        return prompt.strip()

    def extract_json(self, text):
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

    def convert_criteria_to_submission_format(self, criteria):
        """Convert criteria to submission format."""
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

    @torch.no_grad()
    def run_validation(self, epoch: int = None) -> dict:
        """Run validation on val set and compute metrics."""
        if not os.path.exists(self.val_json_path):
            return {"srcc": 0.0, "plcc": 0.0, "level_acc": 0.0, "val_loss": 0.0}

        with open(self.val_json_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)

        val_data = val_data[:self.max_samples]

        results = []

        for item in tqdm(val_data, desc="Validation inference"):
            image_path = self.find_image(item["image_path"])

            if not image_path:
                continue

            prompt = self.build_prompt(item)

            try:
                raw = self._infer_one(image_path, prompt)
                parsed = self.extract_json(raw)

                if parsed:
                    record = {
                        "image_path": item["image_path"],
                        "total_score": parsed.get("total_score"),
                        "criteria": self.convert_criteria_to_submission_format(
                            parsed.get("criteria")
                        ),
                    }
                    results.append(record)
                else:
                    print(f"[WARN] Failed to extract JSON from: {raw[:200] if raw else 'empty'}")
            except Exception as e:
                print(f"[ERROR] Inference failed for {item['image_path']}: {e}")
                continue

        metrics = compute_correlation_metrics(val_data, results)

        # Save validation results
        if self.save_dir is not None and epoch is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"val_{epoch}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return metrics

    def _infer_one(self, image_path, prompt):
        """Run inference on single image."""
        self.model.eval()

        img = self.resize_keep_aspect(image_path, max_size=2048)

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
