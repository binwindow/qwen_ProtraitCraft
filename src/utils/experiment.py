"""Experiment directory setup"""
import os
import json
import torch.distributed as dist


def is_main_process():
    """Check if this is the main process (rank 0) in distributed training."""
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def save_exp_config(exp_name: str, log_dir: str, model_args, data_args, training_args) -> None:
    """Save experiment configuration to log_dir/config.json.

    Args:
        exp_name: Experiment name
        log_dir: Log directory path
        model_args: ModelArguments dataclass
        data_args: DataArguments dataclass
        training_args: TrainingArguments dataclass
    """
    os.makedirs(log_dir, exist_ok=True)

    config = {
        "exp_name": exp_name,
        # Model arguments
        "model_name_or_path": model_args.model_name_or_path,
        "tune_mm_llm": model_args.tune_mm_llm,
        "tune_mm_mlp": model_args.tune_mm_mlp,
        "tune_mm_vision": model_args.tune_mm_vision,
        # Data arguments
        "dataset_path": data_args.dataset_path,
        "model_type": data_args.model_type,
        # Training arguments (all relevant fields)
        "lora_enable": training_args.lora_enable,
        "lora_r": training_args.lora_r,
        "lora_alpha": training_args.lora_alpha,
        "lora_dropout": training_args.lora_dropout,
        "mm_projector_lr": training_args.mm_projector_lr,
        "vision_tower_lr": training_args.vision_tower_lr,
        "train_max_samples": training_args.train_max_samples,
        "data_flatten": training_args.data_flatten,
        "data_packing": training_args.data_packing,
        "max_pixels": training_args.max_pixels,
        "min_pixels": training_args.min_pixels,
        "model_max_length": training_args.model_max_length,
        "cache_dir": training_args.cache_dir,
        # Validation arguments
        "val_json_path": training_args.val_json_path,
        "val_images_path": training_args.val_images_path,
        "val_max_samples": training_args.val_max_samples,
        "val_eval_steps": training_args.val_eval_steps,
        "metric_for_best_model": training_args.metric_for_best_model,
        # Other training args
        "output_dir": training_args.output_dir,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "warmup_ratio": training_args.warmup_ratio,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "bf16": training_args.bf16,
        "fp16": training_args.fp16,
        "gradient_checkpointing": training_args.gradient_checkpointing,
    }

    config_path = os.path.join(log_dir, "config.json")
    if is_main_process():
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Experiment config saved to: {config_path}")


def setup_experiment(config) -> dict:
    """
    Create experiment directory structure and return directory paths.

    Args:
        config: Training config object or dict with save_dir and exp_name

    Returns:
        dict with exp_dir, log_dir, ckpt_dir, sample_dir, plt_fig_dir, test_dir
    """
    if hasattr(config, "save_dir"):
        save_dir = config.save_dir
        exp_name = config.exp_name
    else:
        save_dir = config.get("save_dir", "./outputs")
        exp_name = config.get("exp_name", "default_exp")

    exp_dir = os.path.join(save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    log_dir = os.path.join(exp_dir, "log")
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    sample_dir = os.path.join(exp_dir, "samples")
    plt_fig_dir = os.path.join(exp_dir, "plt_fig")
    test_dir = os.path.join(exp_dir, "test")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(plt_fig_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return {
        "exp_dir": exp_dir,
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
        "sample_dir": sample_dir,
        "plt_fig_dir": plt_fig_dir,
        "test_dir": test_dir,
    }
