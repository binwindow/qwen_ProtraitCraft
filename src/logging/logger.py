"""
Three-layer logging system:
1. SwanLab (wandb-style interface)
2. JSON structured logging (train_metrics.jsonl, val_metrics.jsonl)
3. Python logging (train.log)
"""
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class SwanLabLogger:
    """SwanLab logger with wandb-compatible interface."""

    def __init__(self, exp_name: str, config: Optional[Dict] = None, log_dir: str = "./logs"):
        self.exp_name = exp_name
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.run = None
        self.enabled = False

        try:
            import swanlab
            self.run = swanlab
            self.enabled = True
        except ImportError:
            print("[Logger] swanlab not installed, using fallback logger")
            self.enabled = False

    def init(self, config: Optional[Dict] = None):
        if self.enabled and self.run is not None:
            self.run.init(
                project=self.exp_name,
                name=self.exp_name,
                config=config,
            )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.enabled and self.run is not None:
            if step is not None:
                self.run.log(metrics, step=step)
            else:
                self.run.log(metrics)
        else:
            step_str = f"{step // 1000}K" if step and step >= 1000 else str(step)
            metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
            print(f"[Metrics] step={step_str} | {metrics_str}")

    def finish(self):
        if self.enabled and self.run is not None:
            self.run.finish()


class JSONLogger:
    """JSON-based structured logger for machine learning metrics."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, epoch: Optional[int] = None):
        entry = {"timestamp": datetime.now().isoformat()}
        if step is not None:
            entry["step"] = step
        if epoch is not None:
            entry["epoch"] = epoch
        entry.update(metrics)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


class PythonLogger:
    """Standard Python logging logger."""

    def __init__(self, log_file: str, name: str = "qwen-vl"):
        # Get the named logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        # Do NOT propagate to root logger - we handle everything ourselves
        self.logger.propagate = False

        # Also configure root logger to capture any stray logging.warning/error calls
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        if self.logger.handlers:
            self.logger.handlers.clear()

        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        # Also add handlers to root logger to catch direct logging calls
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)


class LoggerManager:
    """Unified logger coordinating all three logging systems."""

    def __init__(
        self,
        exp_name: str,
        save_dir: str = "./logs",
        config: Optional[Dict] = None,
        use_swanlab: bool = True,
    ):
        self.exp_name = exp_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        train_log_file = os.path.join(self.save_dir, "train_metrics.jsonl")
        val_log_file = os.path.join(self.save_dir, "val_metrics.jsonl")
        python_log_file = os.path.join(self.save_dir, "train.log")

        self.train_logger = JSONLogger(train_log_file)
        self.val_logger = JSONLogger(val_log_file)
        self.python_logger = PythonLogger(python_log_file)

        if use_swanlab:
            self.swanlab_logger = SwanLabLogger(exp_name, config, save_dir)
        else:
            self.swanlab_logger = None

        self.python_logger.info(f"Logger initialized for experiment: {exp_name}")
        self.python_logger.info(f"Log directory: {self.save_dir}")

    def init(self, config: Optional[Dict] = None):
        if self.swanlab_logger:
            self.swanlab_logger.init(config)

    def log_train(self, metrics: Dict[str, Any], step: Optional[int] = None, epoch: Optional[int] = None, console: bool = True):
        self.train_logger.log(metrics, step=step, epoch=epoch)
        if self.swanlab_logger:
            self.swanlab_logger.log(metrics, step=step)
        if console and step is not None:
            step_str = f"{step // 1000}K" if step >= 1000 else str(step)
            metrics_str = " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ])
            self.python_logger.info(f"Step {step_str} | Epoch {epoch} | {metrics_str}")

    def log_val(self, metrics: Dict[str, Any], step: Optional[int] = None, epoch: Optional[int] = None):
        self.val_logger.log(metrics, step=step, epoch=epoch)
        if self.swanlab_logger:
            self.swanlab_logger.log(metrics, step=step)
        if step is not None:
            step_str = f"{step // 1000}K" if step >= 1000 else str(step)
            metrics_str = " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ])
            self.python_logger.info(f"[VAL] Step {step_str} | Epoch {epoch} | {metrics_str}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, epoch: Optional[int] = None):
        """Generic log - defaults to train log."""
        self.log_train(metrics, step, epoch)

    def info(self, msg: str):
        self.python_logger.info(msg)

    def debug(self, msg: str):
        self.python_logger.debug(msg)

    def warning(self, msg: str):
        self.python_logger.warning(msg)

    def error(self, msg: str):
        self.python_logger.error(msg)

    def finish(self):
        if self.swanlab_logger:
            self.swanlab_logger.finish()
