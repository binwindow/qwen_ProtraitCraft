"""Optimizer configuration"""
import torch


def create_optimizer(model, config: dict):
    """Create optimizer based on config."""
    optimizer_name = config.get("optim", "adamw_torch")
    learning_rate = config.get("learning_rate", 1e-5)
    weight_decay = config.get("weight_decay", 0)

    if optimizer_name == "adamw_torch":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        from transformers import AdamW
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    return optimizer


def create_lr_scheduler(optimizer, config: dict, num_training_steps: int):
    """Create learning rate scheduler."""
    scheduler_type = config.get("lr_scheduler_type", "cosine")
    warmup_ratio = config.get("warmup_ratio", 0.03)

    warmup_steps = int(num_training_steps * warmup_ratio)

    if scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: max(1.0 - step / num_training_steps, 0)
        )

    return scheduler
