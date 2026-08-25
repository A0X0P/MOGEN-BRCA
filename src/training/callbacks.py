"""Training loop callbacks: checkpointing, early stopping, and metric logging.

Callbacks are composable — the Trainer accepts a list and fires each hook
at well-defined points in the training loop. Subclass :class:`Callback` and
override the hooks you need.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any

from src.utils.io import save_checkpoint
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Callback(ABC):
    """Base class for training callbacks.

    Override any hook; default implementations are no-ops.
    """

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: Any) -> None:
        """Called at the end of each epoch with aggregated metrics."""

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        """Called after each training batch."""

    def on_best_metric(
        self, epoch: int, metrics: dict[str, float], trainer: Any
    ) -> None:
        """Called when the monitored metric improves."""


class CheckpointCallback(Callback):
    """Saves model checkpoints on metric improvement and at fixed intervals.

    Checkpoint contents: model_state_dict, optimizer_state_dict, epoch,
    config, scheduler_state_dict (if present), and metrics at save time.

    Args:
        save_dir: Directory for checkpoint files.
        monitor: Metric name to watch for best-model saving.
        mode: ``"min"`` or ``"max"`` — whether lower or higher is better.
        save_every_n_epochs: Also save a periodic checkpoint regardless of metric.
    """

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_every_n_epochs: int | None = None,
    ) -> None:
        self._save_dir = Path(save_dir)
        self._monitor = monitor
        self._mode = mode
        self._save_every_n = save_every_n_epochs
        self._best: float | None = None

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: Any) -> None:
        """Save periodic checkpoint if configured."""
        if self._save_every_n and (epoch + 1) % self._save_every_n == 0:
            self._save(epoch, metrics, trainer, suffix=f"epoch_{epoch}")

    def on_best_metric(
        self, epoch: int, metrics: dict[str, float], trainer: Any
    ) -> None:
        """Save best-model checkpoint."""
        self._save(epoch, metrics, trainer, suffix="best")
        logger.info(
            "Best %s=%.5f at epoch %d — checkpoint saved.",
            self._monitor,
            metrics.get(self._monitor, float("nan")),
            epoch,
        )

    @property
    def monitor(self) -> str:
        """The metric being monitored."""
        return self._monitor

    @property
    def mode(self) -> str:
        """Whether to minimise or maximise the monitored metric."""
        return self._mode

    def _save(
        self,
        epoch: int,
        metrics: dict[str, float],
        trainer: Any,
        suffix: str,
    ) -> None:
        state: dict[str, Any] = {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "epoch": epoch,
            "config": trainer.config,
            "metrics": metrics,
        }
        if trainer.scheduler is not None:
            state["scheduler_state_dict"] = trainer.scheduler.state_dict()
        path = self._save_dir / f"checkpoint_{suffix}.pt"
        save_checkpoint(state, path)


class EarlyStoppingCallback(Callback):
    """Halts training when the monitored metric stops improving.

    Args:
        monitor: Metric name to watch (must be in the metrics dict).
        patience: Number of epochs without improvement before stopping.
        mode: ``"min"`` or ``"max"``.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        self._monitor = monitor
        self._patience = patience
        self._mode = mode
        self._min_delta = min_delta
        self._best: float | None = None
        self._wait: int = 0

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: Any) -> None:
        """Check monitored metric and update patience counter."""
        current = metrics.get(self._monitor)
        if current is None:
            return

        if self._is_improvement(current):
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patience:
                logger.info(
                    "Early stopping triggered at epoch %d "
                    "(%s did not improve for %d epochs).",
                    epoch,
                    self._monitor,
                    self._patience,
                )
                trainer.should_stop = True

    def _is_improvement(self, current: float) -> bool:
        if self._best is None:
            return True
        if self._mode == "min":
            return current < self._best - self._min_delta
        return current > self._best + self._min_delta


class LoggingCallback(Callback):
    """Logs epoch metrics and learning rate via the structured logger."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: Any) -> None:
        """Log all metrics for this epoch."""
        lr = trainer.optimizer.param_groups[0]["lr"]
        parts = [f"Epoch {epoch:>3d} | lr={lr:.2e}"]
        for key, value in sorted(metrics.items()):
            parts.append(f"{key}={value:.5f}")
        logger.info(" | ".join(parts))
