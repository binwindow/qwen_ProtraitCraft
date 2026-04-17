"""Logarithmic validation schedule"""
import math


def build_log_schedule(total_steps: int, val_num: int = 50) -> list:
    """
    Generate validation step list with logarithmic spacing (sparse early, dense late).

    Args:
        total_steps: Total number of training steps
        val_num: Target number of validations to perform

    Returns:
        List of step indices where validation should occur
    """
    if val_num <= 0:
        return []
    if total_steps <= 0:
        return []
    if val_num >= total_steps:
        return list(range(total_steps))

    targets = set()
    for k in range(val_num):
        if val_num == 1:
            t = 0.5
        else:
            t = k / (val_num - 1)
        target = (total_steps - 1) * (10 ** t - 1) / 9
        target = round(target)
        target = max(0, min(target, total_steps - 1))
        targets.add(target)

    return sorted(list(targets))
