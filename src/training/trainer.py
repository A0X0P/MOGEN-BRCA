"""Training loop: orchestrates model training over a MultimodalDataset.

The Trainer is config-driven end-to-end — no hyperparameters are hardcoded.
It delegates checkpointing, early stopping, and logging to composable
Callback objects and supports mixed precision, gradient accumulation, and
resume-from-checkpoint.

Every batch carries per-task masks alongside its labels; the masks are passed
straight through to the loss and to the validation metric collector, so a
patient contributes to a task only when that task's target exists.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.data.tasks import CLASSIFICATION_TASKS
from src.evaluation.evaluator import (
    PredictionCollector,
    build_model_inputs,
    extract_labels,
    extract_masks,
)
from src.training.callbacks import Callback
from src.training.losses import MultiTaskLoss
from src.utils.io import load_checkpoint
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


class Trainer:
    """Config-driven training loop for the multimodal cancer model.

    Args:
        model: The model to train (from model_factory).
        train_dataset: Training split as a MultimodalDataset.
        val_dataset: Validation split as a MultimodalDataset.
        loss_fn: MultiTaskLoss (or any callable with the same interface).
        config: Full train.yaml config dict.
        callbacks: Optional list of Callback objects.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        loss_fn: MultiTaskLoss,
        config: dict[str, Any],
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.callbacks: list[Callback] = callbacks or []
        self.should_stop = False

        self.device = torch.device(
            config.get("device")
            if config.get("device") not in (None, "auto", "")
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("learning_rate", 1e-4)),
            weight_decay=float(config.get("weight_decay", 1e-5)),
        )
        self.scheduler = self._build_scheduler()

        batch_size = int(config.get("batch_size", 16))
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_multimodal,
        )
        self._val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_multimodal,
        )

        self._use_amp = (
            bool(config.get("mixed_precision", False)) and self.device.type == "cuda"
        )
        self._scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self._use_amp,
        )
        self._grad_accum = int(config.get("grad_accumulation_steps", 1))
        self._grad_clip = config.get("grad_clip")

        # A single-modality ablation rejects the disabled modality's tensor, so
        # the batch is filtered to whatever this model actually accepts.
        self._modalities = getattr(model, "active_modalities", None)

    def _build_scheduler(
        self,
    ) -> torch.optim.lr_scheduler.LRScheduler | ReduceLROnPlateau | None:
        """Build the learning-rate scheduler declared in the config.

        Returns:
            The configured scheduler, or ``None`` when no scheduler is set.

        Raises:
            ValueError: If the configured scheduler name is unknown.
        """
        spec: Any = self.config.get("scheduler") or {}
        if isinstance(spec, str):
            spec = {"name": spec}

        name = str(spec.get("name", "none")).lower()
        self._scheduler_monitor = str(spec.get("monitor", "val_total"))

        if name in ("", "none"):
            return None

        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(spec.get("t_max", self.config.get("epochs", 50))),
                eta_min=float(spec.get("eta_min", 0.0)),
            )

        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(spec.get("step_size", 10)),
                gamma=float(spec.get("gamma", 0.1)),
            )

        if name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode=str(spec.get("mode", "min")),
                factor=float(spec.get("factor", 0.5)),
                patience=int(spec.get("patience", 5)),
            )

        raise ValueError(
            f"Unknown scheduler '{name}'. Expected one of: "
            "none, cosine, step, plateau."
        )

    def _step_scheduler(self, metrics: dict[str, float]) -> None:
        """Advance the scheduler at the end of an epoch."""
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            monitored = metrics.get(self._scheduler_monitor)
            if monitored is None:
                raise KeyError(
                    f"Scheduler monitors '{self._scheduler_monitor}', which is "
                    f"not among the epoch metrics: {sorted(metrics)}."
                )
            self.scheduler.step(monitored)
            return

        self.scheduler.step()

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            History dict mapping metric names to per-epoch lists.
        """
        set_seed(int(self.config.get("seed", 42)))

        start_epoch = self._maybe_resume()
        epochs = int(self.config.get("epochs", 50))
        history: dict[str, list[float]] = {}

        for epoch in range(start_epoch, epochs):
            self._fire("on_epoch_start", epoch=epoch, trainer=self)

            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._validate_one_epoch()

            metrics = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            for key, value in metrics.items():
                history.setdefault(key, []).append(value)

            self._step_scheduler(metrics)

            self._fire("on_epoch_end", epoch=epoch, metrics=metrics, trainer=self)
            self._maybe_fire_best(epoch, metrics)

            if self.should_stop:
                logger.info("Training stopped at epoch %d.", epoch)
                break

        return history

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""

        self.model.train()

        totals: dict[str, float] = {}
        num_batches = len(self._train_loader)

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self._train_loader):
            inputs = build_model_inputs(batch, self.device, self._modalities)
            labels = extract_labels(batch, self.device)
            masks = extract_masks(batch, self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self._use_amp,
            ):
                output = self.model(**inputs)
                loss_dict = self.loss_fn(output, labels, masks)
                loss = loss_dict["total"] / self._grad_accum

            self._scaler.scale(loss).backward()

            if (batch_idx + 1) % self._grad_accum == 0:
                if self._grad_clip is not None:
                    self._scaler.unscale_(self.optimizer)

                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(self._grad_clip),
                    )

                self._scaler.step(self.optimizer)
                self._scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

            for name, value in loss_dict.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item())

            self._fire(
                "on_batch_end",
                batch_idx=batch_idx,
                loss=float(loss_dict["total"].detach().item()),
                trainer=self,
            )

        return {name: value / max(num_batches, 1) for name, value in totals.items()}

    def _validate_one_epoch(self) -> dict[str, float]:
        """Run one validation epoch, returning losses and task metrics."""

        self.model.eval()

        totals: dict[str, float] = {}
        num_batches = len(self._val_loader)
        collector = PredictionCollector()

        with torch.no_grad():
            for batch in self._val_loader:
                inputs = build_model_inputs(batch, self.device, self._modalities)
                labels = extract_labels(batch, self.device)
                masks = extract_masks(batch, self.device)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self._use_amp,
                ):
                    output = self.model(**inputs)
                    loss_dict = self.loss_fn(output, labels, masks)

                collector.update(output, labels, masks)

                for name, value in loss_dict.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach().item())

        result = {name: value / max(num_batches, 1) for name, value in totals.items()}
        result.update(collector.result().flat_metrics())
        return result

    def _maybe_resume(self) -> int:
        """Load a checkpoint if resume_checkpoint is set in config."""
        ckpt_path = self.config.get("resume_checkpoint")
        if not ckpt_path:
            return 0

        checkpoint = load_checkpoint(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start = int(checkpoint.get("epoch", 0)) + 1
        logger.info("Resumed from checkpoint at epoch %d.", start - 1)
        return start

    def _maybe_fire_best(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Fire checkpoint callback when its monitored metric improves."""

        from src.training.callbacks import CheckpointCallback

        for callback in self.callbacks:
            if not isinstance(callback, CheckpointCallback):
                continue

            current = metrics.get(callback.monitor)

            if current is None:
                continue

            best = getattr(callback, "_best", None)

            if best is None:
                improved = True
            elif callback.mode == "min":
                improved = current < best
            else:
                improved = current > best

            if improved:
                callback._best = current

                callback.on_best_metric(
                    epoch=epoch,
                    metrics=metrics,
                    trainer=self,
                )

    def _fire(self, hook: str, **kwargs: Any) -> None:
        """Call a named hook on every registered callback."""
        for cb in self.callbacks:
            getattr(cb, hook)(**kwargs)


def collate_multimodal(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate matched clinical + genomic samples with their task masks.

    Every sample must carry both active modalities. Targets, by contrast, are
    per-task optional: an absent classification label arrives as
    :data:`~src.data.tasks.IGNORE_INDEX` with ``mask[task] = False``, and an
    absent survival observation arrives as a placeholder ``0.0`` duration with
    ``mask["survival"] = False``. The masks travel with the batch so that the
    loss and the metrics both exclude those rows.

    Args:
        batch: Samples from :class:`~src.data.datasets.multimodal_dataset.MultimodalDataset`.

    Returns:
        Collated batch with ``patient_id``, ``clinical``, ``genomics``,
        ``label``, ``mask`` and ``survival`` entries.

    Raises:
        ValueError: If the batch is empty or a sample lacks a required
            modality or the mask block.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    for sample in batch:
        for key in ("clinical", "genomics", "label", "mask", "survival"):
            if sample.get(key) is None:
                raise ValueError(
                    f"Sample for patient {sample.get('patient_id')} is missing "
                    f"'{key}'."
                )

    mask_tasks = tuple(batch[0]["mask"])

    return {
        "patient_id": [sample["patient_id"] for sample in batch],
        "clinical": {
            "features": torch.stack(
                [sample["clinical"]["features"] for sample in batch]
            )
        },
        "genomics": {
            "features": torch.stack(
                [sample["genomics"]["features"] for sample in batch]
            )
        },
        "label": {
            task: torch.stack([sample["label"][task] for sample in batch]).long()
            for task in CLASSIFICATION_TASKS
        },
        "mask": {
            task: torch.stack([sample["mask"][task] for sample in batch]).bool()
            for task in mask_tasks
        },
        "survival": {
            "duration": torch.stack(
                [sample["survival"]["duration"] for sample in batch]
            ).float(),
            "event": torch.stack(
                [sample["survival"]["event"] for sample in batch]
            ).float(),
        },
    }
