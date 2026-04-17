"""
Training Engine using HuggingFace Accelerate
"""
import math
import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from ..utils.schedule import build_log_schedule


class TrainingEngine:
    """Main training engine with Accelerate integration."""

    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        logger: Any,
        ckpt_manager: Any,
        lr_scheduler: Optional[Any] = None,
        validator: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.ckpt_manager = ckpt_manager
        self.lr_scheduler = lr_scheduler
        self.validator = validator

        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            mixed_precision=config.get("mixed_precision", "bf16"),
            project_dir=config.get("log_dir"),
        )

        self.device = self.accelerator.device

        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader
        )
        if val_loader is not None:
            self.val_loader = self.accelerator.prepare(val_loader)
        if lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(lr_scheduler)

        self.global_step = 0
        self.current_epoch = 0

        num_epochs = config.get("num_train_epochs", 1)
        self.val_schedule = build_log_schedule(
            num_epochs, config.get("val_num", 50)
        )
        self.logger.info(f"Validation schedule (epochs): {self.val_schedule}")

    def train(self):
        """Main training loop."""
        num_epochs = self.config.get("num_train_epochs", 1)

        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Total training steps: {len(self.train_loader) * num_epochs}")
        self.logger.info(f"Device: {self.device}")

        for epoch in range(int(self.current_epoch), int(num_epochs)):
            self.current_epoch = epoch
            self._train_epoch(epoch)

        self.logger.info("Training completed!")
        self.ckpt_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=self.global_step,
            epoch=self.current_epoch,
            lr_scheduler=self.lr_scheduler,
        )
        self.logger.finish()

    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Validate at the start of epoch if scheduled
        if self._should_validate(epoch):
            val_metrics = self._validate(epoch=epoch)
            self.ckpt_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                step=self.global_step,
                epoch=epoch,
                metrics=val_metrics,
                lr_scheduler=self.lr_scheduler,
            )

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            with self.accelerator.accumulate(self.model):
                loss = self._train_step(batch)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get("max_grad_norm", 1.0)
                )

            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            self.global_step += 1
            total_loss += loss.item()
            num_batches += 1
            avg_loss = total_loss / num_batches

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "lr": self._get_lr()
            })

            if self.global_step % self.config.get("logging_steps", 1) == 0:
                self.logger.log_train(
                    {"loss": loss.item(), "avg_loss": avg_loss, "lr": self._get_lr()},
                    step=self.global_step,
                    epoch=epoch
                )

        epoch_avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {epoch} completed | Avg Loss: {epoch_avg_loss:.4f}")

    def _train_step(self, batch: Dict) -> torch.Tensor:
        """Single training step."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        position_ids = batch.get("position_ids")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
        )

        return outputs.loss

    @torch.no_grad()
    def _validate(self, epoch: int = None) -> Dict[str, float]:
        """Run validation."""
        metrics = {}

        # Ensure model is in eval mode before validation
        self.model.eval()

        # If validator is available, run model inference validation
        if self.validator is not None:
            self.logger.info("Running model inference validation...")
            val_metrics = self.validator.run_validation(epoch=epoch)
            metrics.update(val_metrics)
            self.logger.log_val(
                {"srcc": val_metrics.get("srcc", 0), "plcc": val_metrics.get("plcc", 0),
                 "level_acc": val_metrics.get("level_acc", 0)},
                step=self.global_step, epoch=epoch
            )
            self.logger.info(
                f"Validation | SRCC: {val_metrics.get('srcc', 0):.4f} | "
                f"PLCC: {val_metrics.get('plcc', 0):.4f} | "
                f"Level Acc: {val_metrics.get('level_acc', 0):.4f}"
            )

        # If val_loader is available, compute validation loss
        if self.val_loader is not None:
            self.model.eval()
            total_loss = 0
            num_batches = 0

            for batch in tqdm(self.val_loader, desc="Validating loss"):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch.get("attention_mask")
                pixel_values = batch.get("pixel_values")
                image_grid_thw = batch.get("image_grid_thw")
                position_ids = batch.get("position_ids")

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
            metrics["val_loss"] = avg_loss
            self.logger.log_val({"val_loss": avg_loss}, step=self.global_step, epoch=self.current_epoch)
            self.logger.info(f"Validation | Loss: {avg_loss:.4f}")

        self.model.train()
        return metrics

    def _should_validate(self, epoch: int) -> bool:
        """Check if should validate at this epoch."""
        return epoch in self.val_schedule

    def _get_lr(self) -> float:
        """Get current learning rate."""
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_last_lr()[0]
        return self.optimizer.param_groups[0]["lr"]

    def resume(self, checkpoint_path: str):
        """Resume from checkpoint."""
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = self.ckpt_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        self.global_step = checkpoint.get("step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)
        self.logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
