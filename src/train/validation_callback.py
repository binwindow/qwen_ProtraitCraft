"""
Validation and Checkpoint Callback for HuggingFace Trainer
Combines validation and checkpoint saving in a single callback
"""
import os
from typing import Optional

import torch
from transformers import TrainerCallback, TrainerState

from ..evaluation.val_evaluator import ValidationEvaluator
from ..logging.logger import LoggerManager


def _is_main_process():
    """Check if this is the main process (rank 0)."""
    import torch.distributed as dist
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class ValidationAndCheckpointCallback(TrainerCallback):
    """Run validation and save checkpoint with metrics in one callback."""

    def __init__(
        self,
        model,
        processor,
        device: torch.device,
        trainer,
        val_json_path: str,
        images_path: str,
        samples_dir: str,
        output_dir: str,
        log_dir: str,
        exp_name: str,
        val_max_samples: int = 200,
        eval_steps: int = 50,
        metric_for_best_model: str = "srcc",
        top_k: int = 3,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.trainer = trainer
        self.val_json_path = val_json_path
        self.images_path = images_path
        self.samples_dir = samples_dir
        self.output_dir = output_dir
        self.val_max_samples = val_max_samples
        self.eval_steps = eval_steps
        self.metric_for_best_model = metric_for_best_model
        self.top_k = top_k
        self.best_checkpoints = []  # [(metric_value, checkpoint_path), ...]

        # Load existing checkpoints for top-k management (resume training)
        self._load_existing_checkpoints()

        self.validator = ValidationEvaluator(
            model=model,
            processor=processor,
            device=device,
            val_json_path=val_json_path,
            images_path=images_path,
            max_samples=val_max_samples,
            save_dir=samples_dir,
        )

        self.logger_manager = LoggerManager(
            exp_name=exp_name,
            save_dir=log_dir,
            use_swanlab=False,
        )

    def _load_existing_checkpoints(self):
        """Load existing checkpoints from output_dir for top-k management."""
        import re
        if not os.path.exists(self.output_dir):
            return

        # Only main process should manage checkpoints
        if not _is_main_process():
            return

        pattern = rf"checkpoint-(\d+)-{re.escape(self.metric_for_best_model)}([+-]?[\d.]+)"
        for ckpt_dir in os.listdir(self.output_dir):
            match = re.match(pattern, ckpt_dir)
            if match:
                step = int(match.group(1))
                metric_value = float(match.group(2))
                ckpt_path = os.path.join(self.output_dir, ckpt_dir)
                self.best_checkpoints.append((metric_value, ckpt_path, step))
                if _is_main_process():
                    print(f"Loaded existing checkpoint for top-k: {ckpt_dir}")

        # Sort by metric and keep top_k + 1 (for latest)
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        max_to_keep = self.top_k + 1
        if len(self.best_checkpoints) > max_to_keep:
            # Keep latest checkpoint
            latest_step = max(x[2] for x in self.best_checkpoints)
            latest_checkpoint = next(x for x in self.best_checkpoints if x[2] == latest_step)

            # Build new list with latest + top-k-1 others
            others = [x for x in self.best_checkpoints if x[2] != latest_step]
            others = others[:self.top_k - 1] if len(others) >= self.top_k - 1 else others
            new_best = [latest_checkpoint] + others

            # Delete excess
            to_delete = set(x[1] for x in self.best_checkpoints) - set(x[1] for x in new_best)
            self.best_checkpoints = new_best

            for path in to_delete:
                if os.path.exists(path):
                    import shutil
                    shutil.rmtree(path)
                    if _is_main_process():
                        print(f"Removed excess checkpoint: {path}")

    def on_log(self, args, state, control, **kwargs):
        """Log training metrics."""
        logs = kwargs.get("logs", {})
        if logs:
            self.logger_manager.log_train(logs, step=state.global_step, epoch=state.epoch, console=False)

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control,
        **kwargs,
    ):
        """Run validation and save checkpoint at configured step intervals."""
        if self.eval_steps > 0 and state.global_step % self.eval_steps == 0:
            from ..train.trainer_patch import restore_original_attention_class, replace_qwen2_vl_attention_class
            restore_original_attention_class()

            # All processes run validation (val_evaluator handles multi-GPU splitting)
            metrics = self.validator.run_validation(epoch=state.global_step)

            # Reapply flash attention after validation
            replace_qwen2_vl_attention_class()

            # Only main process logs metrics and saves checkpoint
            if state.is_world_process_zero:
                # Log validation metrics
                self.logger_manager.log_val(metrics, step=state.global_step, epoch=state.epoch)

                # 2. Immediately save checkpoint with metrics
                metric_value = metrics.get(self.metric_for_best_model)
                if metric_value is not None:
                    step = state.global_step
                    new_name = f"checkpoint-{step}-{self.metric_for_best_model}{metric_value:.4f}"
                    new_dir = os.path.join(self.output_dir, new_name)

                    # Save model (similar to safe_save_model_for_hf_trainer)
                    self._save_model(new_dir)

                    print(f"Checkpoint saved: {new_name}/")

                    # 3. Manage top-k and last checkpoint
                    self.best_checkpoints.append((metric_value, new_dir, step))
                self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)

                # Keep top_k + 1 (the +1 is for last checkpoint which is also the latest)
                max_to_keep = self.top_k + 1
                if len(self.best_checkpoints) > max_to_keep:
                    # Remove excess checkpoints (all except top-k and the latest one)
                    # The latest checkpoint is the one with highest step in best_checkpoints
                    latest_step = max(x[2] for x in self.best_checkpoints)
                    latest_checkpoint = next(x for x in self.best_checkpoints if x[2] == latest_step)

                    # Rebuild list without the latest and without excess worst
                    new_best = [latest_checkpoint]  # Always keep latest
                    # Add top-k-1 from the remaining (excluding latest)
                    others = [x for x in self.best_checkpoints if x[2] != latest_step]
                    others = others[:self.top_k - 1] if len(others) >= self.top_k - 1 else others
                    new_best.extend(others)

                    # Find which checkpoints to delete
                    to_delete = set(x[1] for x in self.best_checkpoints) - set(x[1] for x in new_best)

                    self.best_checkpoints = new_best

                    for path in to_delete:
                        if os.path.exists(path):
                            import shutil
                            shutil.rmtree(path)
                            print(f"Removed checkpoint: {path}")

    def on_train_end(self, args, state, control, **kwargs):
        """Finish logging on training end."""
        self.logger_manager.finish()

    def _save_model(self, save_dir: str):
        """Save complete checkpoint using trainer methods."""
        os.makedirs(save_dir, exist_ok=True)

        # Sync GPU before saving
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Handle deepspeed vs normal training
        if hasattr(self.trainer, 'deepspeed') and self.trainer.deepspeed:
            # DeepSpeed: use save_model which handles ZeRO correctly
            self.trainer.save_model(save_dir)
        else:
            # Normal training: use _save method
            self.trainer._save(save_dir, state_dict=None)
            # Save optimizer, scheduler, rng_state via trainer methods
            self.trainer._save_optimizer_and_scheduler(save_dir)
            self.trainer._save_rng_state(save_dir)
            self.trainer.state.save_to_json(os.path.join(save_dir, "trainer_state.json"))

        # Save processor/tokenizer files
        self.trainer.processing_class.save_pretrained(save_dir)
