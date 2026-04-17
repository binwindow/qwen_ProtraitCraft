#!/usr/bin/env python3
"""
Qwen-VL Fine-tuning Training Script
Main entry point for training
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed, setup_experiment
from src.logging import LoggerManager
from src.checkpoint import CheckpointManager
from src.model import load_model, apply_flash_attention_patch, save_parameter_stats
from src.data import make_supervised_data_module
from src.train.engine import TrainingEngine
from src.train.optimizer import create_optimizer, create_lr_scheduler
from src.evaluation.val_evaluator import ValidationEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL Fine-tuning")

    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--train_max_samples", type=int, default=None,
                        help="Max training samples to use (None = all)")

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./outputs")

    parser.add_argument("--lora_enable", action="store_true")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--tune_mm_vision", action="store_true")
    parser.add_argument("--tune_mm_mlp", action="store_true")
    parser.add_argument("--tune_mm_llm", action="store_true")

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "no"])

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--val_num", type=int, default=50)
    parser.add_argument("--val_json_path", type=str, default=None,
                        help="Path to validation JSON file")
    parser.add_argument("--val_images_path", type=str, default=None,
                        help="Base path for validation images")
    parser.add_argument("--val_max_samples", type=int, default=200,
                        help="Max samples to validate on")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--model_max_length", type=int, default=8192)

    parser.add_argument("--max_pixels", type=int, default=50176)
    parser.add_argument("--min_pixels", type=int, default=784)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_swanlab", action="store_true", default=True)

    parser.add_argument("--data_flatten", action="store_true")
    parser.add_argument("--data_packing", action="store_true")

    parser.add_argument("--cache_dir", type=str, default=None)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    set_seed(args.seed)

    config = vars(args)
    dirs = setup_experiment(config)
    exp_dir = dirs["exp_dir"]
    log_dir = dirs["log_dir"]
    ckpt_dir = dirs["ckpt_dir"]

    logger = LoggerManager(
        exp_name=args.exp_name,
        save_dir=log_dir,
        config=config,
        use_swanlab=args.use_swanlab,
    )
    logger.init(config)

    logger.info(f"Configuration: {config}")
    logger.info(f"Experiment directory: {exp_dir}")

    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")

    # Only apply flash attention patch when using packed sequences
    if args.data_flatten or args.data_packing:
        apply_flash_attention_patch()
        logger.info("Flash attention patch applied (data_flatten=True)")

    logger.info("Loading model...")
    model, processor, tokenizer, model_type = load_model(
        model_name_or_path=args.model_name_or_path,
        lora_enable=args.lora_enable,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        tune_mm_vision=args.tune_mm_vision,
        tune_mm_mlp=args.tune_mm_mlp,
        tune_mm_llm=args.tune_mm_llm,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        attn_implementation="flash_attention_2",
        cache_dir=args.cache_dir,
    )

    logger.info(f"Model type: {model_type}")

    total_params = model.num_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    param_stats_path = os.path.join(log_dir, "parameter.json")
    save_parameter_stats(model, param_stats_path)
    logger.info(f"Parameter stats saved to {param_stats_path}")

    logger.info("Loading dataset...")
    data_args = {
        "dataset_path": args.dataset_path,
        "dataset_split": args.dataset_split,
        "train_max_samples": args.train_max_samples,
        "model_type": model_type,
        "data_flatten": args.data_flatten,
        "data_packing": args.data_packing,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
    }
    data_module = make_supervised_data_module(processor, data_args)
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]

    logger.info(f"Train dataset size: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4,
    )

    logger.info("Creating optimizer...")
    optimizer = create_optimizer(model, config)

    num_training_steps = len(train_loader) * args.num_train_epochs // args.gradient_accumulation_steps
    lr_scheduler = create_lr_scheduler(optimizer, config, num_training_steps)

    ckpt_manager = CheckpointManager(
        save_dir=ckpt_dir,
        top_k=3,
        metric_optimization="max",
    )

    validator = None
    if args.val_json_path and os.path.exists(args.val_json_path):
        logger.info("Creating validation evaluator...")
        samples_dir = os.path.join(exp_dir, "samples")

        # Load a separate model for validation inference (without flash attention patch)
        # The flash attention patch is incompatible with model.generate()
        logger.info("Loading validation model (without flash attention patch)...")

        # Restore original attention class before loading validation model
        from src.train.trainer_patch import restore_original_attention_class
        restore_original_attention_class()

        from transformers import AutoModelForImageTextToText
        val_model = AutoModelForImageTextToText.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else None,
            attn_implementation="flash_attention_2",
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        validator = ValidationEvaluator(
            model=val_model,
            processor=processor,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            val_json_path=args.val_json_path,
            images_path=args.val_images_path or args.dataset_path,
            max_samples=args.val_max_samples,
            save_dir=samples_dir,
        )
        logger.info(f"Validation evaluator created (max_samples={args.val_max_samples})")
    else:
        logger.info("No validation JSON path provided, skipping validation evaluator")

    engine = TrainingEngine(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=None,
        logger=logger,
        ckpt_manager=ckpt_manager,
        lr_scheduler=lr_scheduler,
        validator=validator,
    )

    last_ckpt_path = ckpt_manager.get_last_checkpoint_path()
    if last_ckpt_path and os.path.exists(last_ckpt_path):
        logger.info(f"Found checkpoint: {last_ckpt_path}, resuming...")
        engine.resume(last_ckpt_path)

    logger.info("Starting training...")
    engine.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
