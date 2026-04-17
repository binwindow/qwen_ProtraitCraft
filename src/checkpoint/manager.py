"""
Checkpoint Management with Top-K Strategy
- Always saves 'last.ckpt'
- Saves top-k checkpoints based on validation metrics
- Auto-cleanup of old checkpoints
"""
import json
import os
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """Manages checkpoint saving with top-k strategy."""

    def __init__(
        self,
        save_dir: str,
        top_k: int = 3,
        metric_optimization: str = "min",
    ):
        self.save_dir = save_dir
        self.top_k = top_k
        self.metric_optimization = metric_optimization

        os.makedirs(save_dir, exist_ok=True)

        self.best_checkpoints = []
        self.last_checkpoint_path = os.path.join(save_dir, "last.ckpt")
        self.info_path = os.path.join(save_dir, "checkpoint_info.json")

        self._load_existing_info()

    def _load_existing_info(self):
        """Load existing checkpoint metadata."""
        if os.path.exists(self.info_path):
            try:
                with open(self.info_path, "r") as f:
                    data = json.load(f)
                    self.best_checkpoints = [
                        (item["metric"], item["path"])
                        for item in data.get("best_checkpoints", [])
                        if item.get("metric") is not None
                    ]
                    self.best_checkpoints.sort(
                        key=lambda x: x[0] if self.metric_optimization == "min" else -x[0]
                    )
            except Exception:
                pass

    def _save_info(self):
        """Save checkpoint metadata."""
        data = {
            "best_checkpoints": [
                {"metric": metric, "path": path} for metric, path in self.best_checkpoints
            ]
        }
        with open(self.info_path, "w") as f:
            json.dump(data, f, indent=2)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> str:
        """
        Save checkpoint with top-k strategy.

        Returns:
            Path to the saved checkpoint
        """
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        lr_scheduler_state = lr_scheduler.state_dict() if lr_scheduler else None

        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "lr_scheduler_state": lr_scheduler_state,
            "metrics": metrics or {},
        }

        last_path = os.path.join(self.save_dir, "last.ckpt")
        torch.save(checkpoint, last_path)

        if metrics is not None:
            # For srcc (higher is better), use metric_optimization="max"
            # For val_loss (lower is better), use metric_optimization="min"
            primary_metric = metrics.get("srcc") or metrics.get("val_loss") or metrics.get("clipscore")
            if primary_metric is not None:
                self._save_best_checkpoint(checkpoint, primary_metric, metrics)

        return last_path

    def _save_best_checkpoint(
        self,
        checkpoint: dict,
        primary_metric: float,
        full_metrics: Dict[str, Any],
    ) -> str:
        """Save checkpoint if it's one of the top-k best."""
        step = checkpoint["step"]
        metric_str = f"{primary_metric:.4f}"
        filename = f"topk_step{step}_{metric_str}.ckpt"
        filepath = os.path.join(self.save_dir, filename)

        should_save = False
        if len(self.best_checkpoints) < self.top_k:
            should_save = True
        else:
            if self.metric_optimization == "min":
                worst_metric = self.best_checkpoints[-1][0]
                should_save = primary_metric < worst_metric
            else:
                worst_metric = self.best_checkpoints[-1][0]
                should_save = primary_metric > worst_metric

        if should_save:
            torch.save(checkpoint, filepath)
            self.best_checkpoints.append((primary_metric, filepath))
            if self.metric_optimization == "min":
                self.best_checkpoints.sort(key=lambda x: x[0])
            else:
                self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)

            if len(self.best_checkpoints) > self.top_k:
                _, worst_path = self.best_checkpoints.pop(-1)
                if os.path.exists(worst_path) and "topk_" in worst_path:
                    os.remove(worst_path)

            self._save_info()

        return filepath

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> dict:
        """Load checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.last_checkpoint_path

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            device = next(model.parameters()).device

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])

        if optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        if lr_scheduler and "lr_scheduler_state" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

        return checkpoint

    def get_last_checkpoint_path(self) -> Optional[str]:
        """Get path to last checkpoint."""
        if os.path.exists(self.last_checkpoint_path):
            return self.last_checkpoint_path
        return None

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint by metric."""
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]
        return self.get_last_checkpoint_path()
