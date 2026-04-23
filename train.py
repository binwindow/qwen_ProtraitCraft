#!/usr/bin/env python3
"""
Qwen-VL Fine-tuning Training Script
Aligned with upstream: qwenvl/train/train_qwen.py
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import TrainerCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train.trainer_patch import replace_qwen2_vl_attention_class, create_optimizer
from src.train.validation_callback import ValidationAndCheckpointCallback
from src.train.logging_callback import LoggingCallback
from src.data import make_supervised_data_module
from src.logging.logger import LoggerManager
from src.utils.experiment import save_exp_config, is_main_process


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_path: str = field(default="")
    model_type: str = field(default="qwen3vl")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Custom TrainingArguments with LoRA and data arguments."""

    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    mm_projector_lr: Optional[float] = field(default=None)
    vision_tower_lr: Optional[float] = field(default=None)
    train_max_samples: Optional[int] = field(default=None)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    max_pixels: int = field(default=50176)
    min_pixels: int = field(default=784)
    model_max_length: int = field(default=8192)
    cache_dir: Optional[str] = field(default=None)
    # Validation arguments
    val_json_path: Optional[str] = field(default=None)
    val_images_path: Optional[str] = field(default=None)
    val_max_samples: int = field(default=200)
    val_eval_steps: int = field(default=50)
    metric_for_best_model: str = field(default="srcc")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train():
    parser = transformers.HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments,
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create structured output directory
    exp_name = Path(training_args.output_dir).name
    base_dir = Path(training_args.output_dir).parent
    output_dir = base_dir / exp_name

    ckpt_dir = output_dir / "ckpt"
    log_dir = output_dir / "log"
    samples_dir = output_dir / "samples"
    plt_fig_dir = output_dir / "plt_fig"
    test_dir = output_dir / "test"

    for d in [ckpt_dir, log_dir, samples_dir, plt_fig_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save experiment config to log_dir
    save_exp_config(exp_name, str(log_dir), model_args, data_args, training_args)

    # Trainer saves checkpoints to ckpt_dir
    training_args.output_dir = str(ckpt_dir)

    # Apply flash attention patch only when needed (same as upstream)
    if training_args.data_flatten or training_args.data_packing:
        replace_qwen2_vl_attention_class()

    # Model loading (same pattern as upstream)
    model_path = model_args.model_name_or_path
    model_path_lower = model_path.lower()
    model_basename = Path(model_path.rstrip("/")).name.lower()

    # Check for MoE variant: model name ends with "a" (e.g., Qwen3-VL-4B-A-Instruct)
    is_moe = "qwen3" in model_path_lower and (
        "_a_" in model_basename or model_basename.endswith("-a") or model_basename.endswith("_a")
    )
    if is_moe:
        from transformers import Qwen3VLMoeForConditionalGeneration
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model_type = "qwen3vl"
    elif "qwen3" in model_path_lower:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model_type = "qwen3vl"
    elif "qwen2.5" in model_path_lower:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model_type = "qwen2.5vl"
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model_type = "qwen2vl"

    if is_main_process():
        print(f'Loaded model: {model_path}, class: {model.__class__.__name__}')
    model.config.use_cache = False

    # Gradient checkpointing (same as upstream)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Processor and tokenizer
    processor = transformers.AutoProcessor.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # LoRA setup (same as upstream)
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        if is_main_process():
            print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    # Data module
    data_module_kwargs = {
        "dataset_path": data_args.dataset_path,
        "dataset_split": "train",
        "train_max_samples": training_args.train_max_samples,
        "model_type": model_type,
        "data_flatten": training_args.data_flatten,
        "data_packing": training_args.data_packing,
        "max_pixels": training_args.max_pixels,
        "min_pixels": training_args.min_pixels,
    }
    data_module = make_supervised_data_module(processor, data_module_kwargs)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect validation paths
    if training_args.val_json_path is None:
        training_args.val_json_path = os.path.join(data_args.dataset_path, "track_1_val.json")
    if training_args.val_images_path is None:
        training_args.val_images_path = data_args.dataset_path

    # Create Trainer
    trainer = transformers.Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    # Validation callback (after trainer creation to pass trainer reference)
    validation_callback = ValidationAndCheckpointCallback(
        model=model,
        processor=processor,
        device=device,
        trainer=trainer,
        val_json_path=training_args.val_json_path,
        images_path=training_args.val_images_path,
        samples_dir=str(samples_dir),
        output_dir=str(ckpt_dir),
        log_dir=str(log_dir),
        exp_name=exp_name,
        val_max_samples=training_args.val_max_samples,
        eval_steps=training_args.val_eval_steps,
        metric_for_best_model=training_args.metric_for_best_model,
    )
    trainer.add_callback(validation_callback)

    # Check for checkpoint to resume
    resume_from = None
    if ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("checkpoint-*"))
        if checkpoints:
            # Get the checkpoint with highest step
            latest_ckpt = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
            resume_from = str(latest_ckpt)
            if is_main_process():
                print(f"Found checkpoint to resume: {resume_from}")

    # Train
    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model (only on main process)
    if is_main_process():
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    trainer.save_state()


if __name__ == "__main__":
    train()
