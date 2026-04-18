"""
Qwen-VL Model Loading and Configuration
Based on original project: qwenvl/train/train_qwen.py and qwenvl/train/trainer.py
"""
from typing import Optional, Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer

from ..train.trainer_patch import replace_qwen2_vl_attention_class

apply_flash_attention_patch = replace_qwen2_vl_attention_class


def _extract_model_name_from_path(model_path: str) -> str:
    """Extract model name from path for architecture detection.

    Handles both direct model names and HuggingFace cache paths:
    - "Qwen/Qwen3-VL-4B-Instruct" -> "qwen3-vl-4b-instruct"
    - ".../models--Qwen--Qwen3-VL-4B-Instruct/snapshots/..." -> "qwen3-vl-4b-instruct"
    """
    if "huggingface" in model_path.lower() and "models--" in model_path:
        for part in model_path.replace("\\", "/").split("/"):
            if part.startswith("models--"):
                return part.replace("models--", "").replace("--", "/")
    from pathlib import Path
    return Path(model_path.rstrip("/")).name


def load_model(
    model_name_or_path: str,
    lora_enable: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    tune_mm_vision: bool = False,
    tune_mm_mlp: bool = False,
    tune_mm_llm: bool = False,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    attn_implementation: str = "flash_attention_2",
    cache_dir: Optional[str] = None,
) -> tuple:
    """
    Load Qwen-VL model with optional LoRA configuration.

    Args:
        model_name_or_path: HuggingFace model name or path
        lora_enable: Whether to enable LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        tune_mm_vision: Whether to tune vision encoder
        tune_mm_mlp: Whether to tune vision MLP
        tune_mm_llm: Whether to tune LLM
        bf16: Whether to use bfloat16
        gradient_checkpointing: Whether to enable gradient checkpointing
        attn_implementation: Attention implementation
        cache_dir: Cache directory

    Returns:
        tuple: (model, processor, tokenizer)
    """
    from transformers import Qwen2VLForConditionalGeneration
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import Qwen3VLMoeForConditionalGeneration

    model_name_lower = model_name_or_path.lower()
    model_basename = _extract_model_name_from_path(model_name_or_path).lower()

    if "qwen3" in model_name_lower and "a" in model_basename:
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if bf16 else None,
        )
        model_type = "qwen3vl"
    elif "qwen3" in model_name_lower:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if bf16 else None,
        )
        model_type = "qwen3vl"
    elif "qwen2.5" in model_name_lower:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if bf16 else None,
        )
        model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if bf16 else None,
        )
        model_type = "qwen2vl"

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    if gradient_checkpointing:
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        _set_model_trainable(
            model, model_type,
            tune_mm_vision=tune_mm_vision,
            tune_mm_mlp=tune_mm_mlp,
            tune_mm_llm=tune_mm_llm
        )

    return model, processor, tokenizer, model_type


def _set_model_trainable(model, model_type, tune_mm_vision=False, tune_mm_mlp=False, tune_mm_llm=False):
    """Set model trainable parameters based on tuning strategy."""
    # Qwen3VL has model.model.visual, Qwen2VL has model.visual
    if model_type == "qwen3vl":
        visual = model.model.visual
        language_model = model.model.language_model
    else:
        visual = model.visual
        language_model = model.language_model

    if tune_mm_vision:
        for p in visual.parameters():
            p.requires_grad = True
    else:
        for p in visual.parameters():
            p.requires_grad = False

    if tune_mm_mlp:
        for p in visual.merger.parameters():
            p.requires_grad = True
    else:
        for p in visual.merger.parameters():
            p.requires_grad = False

    if tune_mm_llm:
        for p in language_model.parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for p in language_model.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def setup_lora(model, lora_r=64, lora_alpha=128, lora_dropout=0.05):
    """Setup LoRA on an already loaded model."""
    from peft import LoraConfig, get_peft_model, TaskType

    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


def apply_flash_attention_patch():
    """Apply flash attention patch for Qwen-VL models."""
    replace_qwen2_vl_attention_class()


def get_trainable_parameters(model) -> tuple:
    """Get parameter statistics."""
    total_params = 0
    trainable_params = 0

    for p in model.parameters():
        numel = p.numel()
        total_params += numel
        if p.requires_grad:
            trainable_params += numel

    return total_params, trainable_params


def print_trainable_parameters(model):
    """Print trainable parameter statistics."""
    total_params, trainable_params = get_trainable_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")


def save_parameter_stats(model, save_path: str):
    """Save parameter statistics to JSON file."""
    import json
    from collections import defaultdict

    modules_stats = defaultdict(lambda: {"count": 0, "total_numel": 0})
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        module_name = ".".join(name.split(".")[:2]) if "." in name else name
        modules_stats[module_name]["count"] += 1
        modules_stats[module_name]["total_numel"] += param.numel()

        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
        "modules": [
            {
                "name": name,
                "numel": info["total_numel"],
                "param_count": info["count"],
            }
            for name, info in sorted(modules_stats.items())
        ],
    }

    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats
