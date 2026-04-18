"""
Logging Callback for HuggingFace Trainer
Integrates LoggerManager with transformers.Trainer
"""
from transformers import TrainerCallback, TrainerState

from ..logging.logger import LoggerManager


class LoggingCallback(TrainerCallback):
    """Callback that logs metrics using LoggerManager."""

    def __init__(
        self,
        log_dir: str,
        exp_name: str,
        logging_steps: int = 1,
    ):
        self.logger_manager = LoggerManager(
            exp_name=exp_name,
            save_dir=log_dir,
            use_swanlab=False,  # Disable swanlab for now
        )

    def on_log(self, args, state: TrainerState, control, **kwargs):
        """Log metrics."""
        logs = kwargs.get("logs", {})
        if logs:
            self.logger_manager.log_train(logs, step=state.global_step, epoch=state.epoch)

    def on_step_end(self, args, state: TrainerState, control, **kwargs):
        """Log at step end."""
        pass

    def on_train_end(self, args, state: TrainerState, control, **kwargs):
        """Finish logging."""
        self.logger_manager.finish()
