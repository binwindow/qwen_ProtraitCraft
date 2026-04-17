"""Experiment directory setup"""
import os


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
