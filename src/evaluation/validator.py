"""
Validator for model evaluation
"""
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional


class Validator:
    """Validator class for model evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.logger = logger

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            image_grid_thw = batch.get("image_grid_thw")
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(self.device)
            position_ids = batch.get("position_ids")
            if position_ids is not None:
                position_ids = position_ids.to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
            )

            total_loss += outputs.loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        metrics = {"val_loss": avg_loss}

        if self.logger:
            self.logger.info(f"Validation | Loss: {avg_loss:.4f}")

        self.model.train()
        return metrics

    @torch.no_grad()
    def compute_clipscore(self) -> Dict[str, float]:
        """Compute CLIPScore for image-text matching."""
        try:
            import clip
            from PIL import Image
            import torch.nn.functional as F

            model, preprocess = clip.load("ViT-B/32", device=self.device)
            model.eval()
        except ImportError:
            return {"clipscore": 0.0}

        total_score = 0
        num_samples = 0

        for batch in self.val_loader:
            images = batch.get("pixel_values")
            if images is None:
                continue

            text = batch.get("text", [""] * images.size(0))

            for i, img in enumerate(images):
                img_pil = Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"))
                image_input = preprocess(img_pil).unsqueeze(0).to(self.device)
                text_input = clip.tokenize([text[i]]).to(self.device)

                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_input)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).mean().item()
                total_score += similarity
                num_samples += 1

        clipscore = total_score / num_samples if num_samples > 0 else 0.0
        return {"clipscore": clipscore}
