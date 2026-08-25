"""Mask-aware evaluator for the two-modality TCGA-BRCA model.

The evaluator is read-only with respect to model weights. It accepts a trained
model and a DataLoader using the training collate function, aggregates
predictions across batches, and returns a structured
:class:`EvaluationResult`.

Masking contract
----------------
Metrics for a task are computed over the patients whose mask for that task is
``True`` only. A task with no usable labels in the evaluated split reports no
metrics rather than a fabricated score (CLAUDE.md section 21).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.tasks import (
    CLASSIFICATION_TASKS,
    RECEPTOR_TASKS,
    RISK_SCORE_KEY,
    SURVIVAL_TASK,
    TASK_LOGIT_KEYS,
)
from src.evaluation import metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: The two active modality inputs, in the order the fusion stage expects them.
#: Single-modality ablations use a subset of these.
ACTIVE_MODALITIES: tuple[str, ...] = ("clinical", "genomics")


@dataclass
class EvaluationResult:
    """Structured output from a full evaluation pass.

    Attributes:
        classification: Per-task metric mappings, keyed by task name.
        survival: Survival metrics, or ``None`` when the survival task has no
            usable observations or the model has no survival head.
        calibration: Per-task calibration metrics for the binary tasks.
        task_counts: Number of masked-in patients per task.
        n_samples: Total number of patients seen, before masking.
    """

    classification: dict[str, dict[str, float]] = field(default_factory=dict)
    survival: Optional[dict[str, float]] = None
    calibration: dict[str, dict[str, float]] = field(default_factory=dict)
    task_counts: dict[str, int] = field(default_factory=dict)
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the result."""
        return {
            "classification": self.classification,
            "survival": self.survival,
            "calibration": self.calibration,
            "task_counts": self.task_counts,
            "n_samples": self.n_samples,
        }

    def flat_metrics(self) -> dict[str, float]:
        """Flatten to ``<task>_<metric>`` keys, for logging and monitoring."""
        flat: dict[str, float] = {}
        for task, task_metrics in self.classification.items():
            for name, value in task_metrics.items():
                flat[f"{task}_{name}"] = value
        for task, task_metrics in self.calibration.items():
            for name, value in task_metrics.items():
                flat[f"{task}_{name}"] = value
        for name, value in (self.survival or {}).items():
            flat[f"{SURVIVAL_TASK}_{name}"] = value
        return flat


class PredictionCollector:
    """Accumulates masked-in predictions across batches.

    Shared by the evaluator and the training loop so that validation metrics
    and test metrics are computed by exactly the same code path.
    """

    def __init__(self) -> None:
        self._logits: dict[str, list[torch.Tensor]] = {
            task: [] for task in CLASSIFICATION_TASKS
        }
        self._labels: dict[str, list[torch.Tensor]] = {
            task: [] for task in CLASSIFICATION_TASKS
        }
        self._risk: list[torch.Tensor] = []
        self._durations: list[torch.Tensor] = []
        self._events: list[torch.Tensor] = []
        self.n_samples = 0

    def update(
        self,
        output: Mapping[str, torch.Tensor],
        labels: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
    ) -> None:
        """Record one batch, keeping only the rows each task's mask selects.

        Args:
            output: Model forward output.
            labels: Flat label mapping (``subtype``/``er``/``pr``/``her2``/
                ``duration``/``event``).
            masks: Per-task boolean masks.
        """
        batch_size = next(iter(output.values())).shape[0]
        self.n_samples += int(batch_size)

        for task in CLASSIFICATION_TASKS:
            logit_key = TASK_LOGIT_KEYS[task]
            if logit_key not in output or task not in labels:
                continue

            valid = _as_bool_mask(masks.get(task), batch_size, output[logit_key].device)
            if not bool(valid.any()):
                continue

            self._logits[task].append(output[logit_key][valid].detach().float().cpu())
            self._labels[task].append(labels[task][valid].detach().long().cpu())

        self._update_survival(output, labels, masks, batch_size)

    def _update_survival(
        self,
        output: Mapping[str, torch.Tensor],
        labels: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        """Record the survival rows this batch contributes."""
        if RISK_SCORE_KEY not in output:
            return
        if "duration" not in labels or "event" not in labels:
            return

        risk = output[RISK_SCORE_KEY].reshape(-1)
        valid = _as_bool_mask(masks.get(SURVIVAL_TASK), batch_size, risk.device)
        if not bool(valid.any()):
            return

        self._risk.append(risk[valid].detach().float().cpu())
        self._durations.append(labels["duration"].reshape(-1)[valid].detach().cpu())
        self._events.append(labels["event"].reshape(-1)[valid].detach().cpu())

    def result(self) -> EvaluationResult:
        """Compute all applicable metrics from the accumulated predictions."""
        result = EvaluationResult(n_samples=self.n_samples)

        for task in CLASSIFICATION_TASKS:
            if not self._logits[task]:
                result.task_counts[task] = 0
                continue

            logits = torch.cat(self._logits[task], dim=0)
            labels = torch.cat(self._labels[task], dim=0).numpy()
            probs = torch.softmax(logits, dim=-1).numpy()
            preds = logits.argmax(dim=-1).numpy()

            result.task_counts[task] = int(labels.shape[0])
            result.classification[task] = _classification_metrics(labels, preds, probs)

            if task in RECEPTOR_TASKS:
                result.calibration[task] = {
                    "brier_score": metrics.brier_score(labels, probs[:, 1])
                }

        result.survival = self._survival_metrics()
        result.task_counts[SURVIVAL_TASK] = sum(t.numel() for t in self._risk)
        return result

    def _survival_metrics(self) -> Optional[dict[str, float]]:
        """Compute the concordance index, or ``None`` without observations."""
        if not self._risk:
            return None

        risk = torch.cat(self._risk, dim=0).numpy()
        durations = torch.cat(self._durations, dim=0).numpy()
        events = torch.cat(self._events, dim=0).numpy()

        return {
            "concordance_index": metrics.concordance_index(risk, durations, events),
            "n_events": float(events.sum()),
        }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> EvaluationResult:
    """Run a full evaluation pass over a dataloader.

    Args:
        model: Trained model. Set to eval mode by this function.
        dataloader: DataLoader using the training collate function, yielding
            ``clinical``/``genomics``/``label``/``mask``/``survival`` entries.
        device: Target device. Inferred from the model parameters when omitted.

    Returns:
        The aggregated :class:`EvaluationResult`.
    """
    if device is None:
        device = next(model.parameters()).device

    modalities = getattr(model, "active_modalities", None)

    model.eval()
    collector = PredictionCollector()

    with torch.no_grad():
        for batch in dataloader:
            output = model(**build_model_inputs(batch, device, modalities))
            collector.update(
                output,
                extract_labels(batch, device),
                extract_masks(batch, device),
            )

    result = collector.result()
    logger.info(
        "Evaluation complete: %d patients, task counts %s.",
        result.n_samples,
        result.task_counts,
    )
    return result


def build_model_inputs(
    batch: Mapping[str, Any],
    device: torch.device,
    modalities: Optional[Sequence[str]] = None,
) -> dict[str, torch.Tensor]:
    """Extract the modality tensors a model expects from a collated batch.

    Args:
        batch: Collated batch from the training collate function.
        device: Device to move the tensors to.
        modalities: Which modalities to supply. Defaults to both active
            modalities. Single-modality ablations must pass their model's
            :attr:`~src.models.model_factory.MultimodalCancerModel.active_modalities`,
            because the model rejects a disabled modality's tensor rather than
            ignoring it.

    Returns:
        Keyword arguments for the model forward pass.

    Raises:
        KeyError: If a requested modality is absent from the batch.
        ValueError: If ``modalities`` names something outside the two active
            modalities, or is empty.
    """
    requested = tuple(modalities) if modalities is not None else ACTIVE_MODALITIES

    unsupported = set(requested) - set(ACTIVE_MODALITIES)
    if unsupported:
        raise ValueError(
            f"Unsupported modalities: {sorted(unsupported)}. "
            f"Expected a subset of {list(ACTIVE_MODALITIES)}."
        )
    if not requested:
        raise ValueError("At least one modality must be requested.")

    for modality in requested:
        if modality not in batch:
            raise KeyError(f"Batch is missing the '{modality}' modality.")

    return {
        modality: batch[modality]["features"].to(device) for modality in requested
    }


def extract_labels(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    """Flatten a collated batch's targets onto the loss/metric key names."""
    labels = {
        task: batch["label"][task].to(device) for task in CLASSIFICATION_TASKS
    }
    labels["duration"] = batch["survival"]["duration"].to(device)
    labels["event"] = batch["survival"]["event"].to(device)
    return labels


def extract_masks(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move a collated batch's per-task masks to the target device."""
    return {task: mask.to(device) for task, mask in batch["mask"].items()}


def _classification_metrics(
    labels: np.ndarray, preds: np.ndarray, probs: np.ndarray
) -> dict[str, float]:
    """Compute classification metrics for one task's masked-in rows."""
    result: dict[str, float] = {
        "accuracy": metrics.accuracy(labels, preds),
        "precision": metrics.precision(labels, preds),
        "recall": metrics.recall(labels, preds),
        "f1": metrics.f1(labels, preds),
    }

    # ROC-AUC and PR-AUC are undefined when a split lacks a class entirely;
    # that is reported as an absent metric rather than a substituted value.
    for name, fn in (("roc_auc", metrics.roc_auc), ("pr_auc", metrics.pr_auc)):
        try:
            result[name] = fn(labels, probs)
        except ValueError as exc:
            logger.debug("Skipping %s: %s", name, exc)

    return result


def _as_bool_mask(
    mask: Optional[torch.Tensor], batch_size: int, device: torch.device
) -> torch.Tensor:
    """Return a boolean mask, defaulting to all-valid when none is supplied."""
    if mask is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return mask.bool().reshape(-1).to(device)
