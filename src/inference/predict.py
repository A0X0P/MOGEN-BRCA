"""Inference entry point: load a trained model and score a Patient record.

Inference must reproduce the training-time preprocessing exactly, so the
train-fold statistics (clinical age mean/std and gene-wise mean/std) are read
from the checkpoint's embedded config rather than refitted here. Refitting on
the inference cohort would leak that cohort's distribution into its own
features (CLAUDE.md sections 11 and 12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from src.data.datasets.multimodal_dataset import MultimodalDataset
from src.data.pam50 import PAM50_GENES
from src.data.schema.patient import Patient
from src.data.tasks import (
    CLASSIFICATION_TASKS,
    RISK_SCORE_KEY,
    TASK_CLASS_LABELS,
    TASK_LOGIT_KEYS,
)
from src.models.model_factory import MultimodalCancerModel, build_model
from src.utils.io import load_checkpoint, load_yaml
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TaskPrediction:
    """One classification task's prediction for one patient.

    Attributes:
        task: Task name (``"subtype"``, ``"er"``, ``"pr"``, ``"her2"``).
        probabilities: Softmax probabilities, indexed by class.
        predicted_index: Argmax class index.
        predicted_label: Human-readable label for ``predicted_index``.
    """

    task: str
    probabilities: list[float]
    predicted_index: int
    predicted_label: str


@dataclass
class PredictionResult:
    """Structured output from a single inference call.

    Attributes:
        patient_id: ID of the patient that was scored.
        tasks: Per-task predictions, keyed by task name.
        risk_score: DeepSurv log-risk score, or ``None`` without a survival
            head.
        modalities_used: Which modalities contributed to this prediction.
    """

    patient_id: str
    tasks: dict[str, TaskPrediction] = field(default_factory=dict)
    risk_score: Optional[float] = None
    modalities_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the result."""
        return {
            "patient_id": self.patient_id,
            "tasks": {
                name: {
                    "probabilities": task.probabilities,
                    "predicted_index": task.predicted_index,
                    "predicted_label": task.predicted_label,
                }
                for name, task in self.tasks.items()
            },
            "risk_score": self.risk_score,
            "modalities_used": self.modalities_used,
        }


@dataclass
class InferenceArtifacts:
    """A loaded model together with the preprocessing it was trained with.

    Attributes:
        model: The model in eval mode, already moved to ``device``.
        config: The configuration the checkpoint was produced with.
        device: Device the model lives on.
        normalization_stats: Clinical statistics fitted on the training fold,
            or ``None`` when the run recorded none.
        gene_standardization: Gene-wise statistics fitted on the training fold,
            or ``None`` when the run recorded none.
        gene_order: Canonical gene ordering used at training time.
    """

    model: MultimodalCancerModel
    config: dict[str, Any]
    device: torch.device
    normalization_stats: Optional[dict[str, dict[str, float]]] = None
    gene_standardization: Optional[dict[str, list[float]]] = None
    gene_order: tuple[str, ...] = PAM50_GENES


def resolve_device(config: dict[str, Any]) -> torch.device:
    """Resolve the inference device from config, per the project convention."""
    cfg_device: str | None = config.get("device")
    if cfg_device and cfg_device not in ("auto", ""):
        return torch.device(cfg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
) -> InferenceArtifacts:
    """Rebuild the model from config and load trained weights.

    Args:
        checkpoint_path: Path to a checkpoint saved by the training loop.
        config_path: Optional path to the config YAML. When omitted, the
            config embedded in the checkpoint is used, which is the
            authoritative record of the run.

    Returns:
        The loaded :class:`InferenceArtifacts`.

    Raises:
        FileNotFoundError: If a supplied path does not exist.
        KeyError: If the checkpoint has no ``"model_state_dict"``.
        ValueError: If no config is available from either source.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} has no 'model_state_dict' key. "
            f"Found keys: {sorted(checkpoint)}"
        )

    embedded: dict[str, Any] = checkpoint.get("config") or {}
    config = dict(load_yaml(config_path)) if config_path else dict(embedded)
    if not config:
        raise ValueError(
            f"No configuration available: {checkpoint_path} embeds none and no "
            "config_path was supplied."
        )

    device = resolve_device(config)
    model: MultimodalCancerModel = build_model(config.get("model", config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocessing statistics always come from the run that produced the
    # weights, never from the inference-time cohort.
    data_cfg: dict[str, Any] = (embedded or config).get("data", {})
    gene_order = data_cfg.get("gene_order")

    logger.info(
        "Model loaded on %s (checkpoint epoch=%s, clinical_stats=%s, gene_stats=%s).",
        device,
        checkpoint.get("epoch", "?"),
        data_cfg.get("normalization_stats") is not None,
        data_cfg.get("gene_standardization") is not None,
    )

    return InferenceArtifacts(
        model=model,
        config=config,
        device=device,
        normalization_stats=data_cfg.get("normalization_stats"),
        gene_standardization=data_cfg.get("gene_standardization"),
        gene_order=tuple(gene_order) if gene_order else PAM50_GENES,
    )


def predict(
    patient: Patient,
    artifacts: InferenceArtifacts,
) -> PredictionResult:
    """Score a single patient across all five tasks.

    Args:
        patient: A validated Patient carrying both active modalities.
        artifacts: Result of :func:`load_model`.

    Returns:
        A :class:`PredictionResult` with per-task predictions and the optional
        risk score.

    Raises:
        TypeError: If ``patient`` is not a Patient instance.
        ValueError: If the patient lacks a required modality.
    """
    if not isinstance(patient, Patient):
        raise TypeError(f"Expected a Patient instance, got {type(patient).__name__}.")

    modalities_used = patient.available_modalities()
    logger.info(
        "Predicting for patient_id=%s | modalities=%s.",
        patient.patient_id,
        modalities_used,
    )

    inputs = _patient_to_inputs(patient, artifacts)

    with torch.no_grad():
        output: dict[str, torch.Tensor] = artifacts.model(**inputs)

    result = PredictionResult(
        patient_id=patient.patient_id,
        modalities_used=modalities_used,
    )

    for task in CLASSIFICATION_TASKS:
        logit_key = TASK_LOGIT_KEYS[task]
        if logit_key not in output:
            continue
        result.tasks[task] = _decode_task(task, output[logit_key])

    if RISK_SCORE_KEY in output:
        result.risk_score = float(output[RISK_SCORE_KEY].reshape(-1)[0].item())

    return result


def _patient_to_inputs(
    patient: Patient,
    artifacts: InferenceArtifacts,
) -> dict[str, torch.Tensor]:
    """Build batched model inputs for one patient via the dataset encoders.

    Reuses :class:`~src.data.datasets.multimodal_dataset.MultimodalDataset` so
    that no encoding logic is duplicated between training and inference.
    """
    dataset = MultimodalDataset(
        patients=[patient],
        normalization_stats=artifacts.normalization_stats,
        gene_standardization=artifacts.gene_standardization,
        gene_order=artifacts.gene_order,
    )

    sample: dict[str, Any] = dataset[0]
    device = artifacts.device

    return {
        "clinical": sample["clinical"]["features"].unsqueeze(0).to(device),
        "genomics": sample["genomics"]["features"].unsqueeze(0).to(device),
    }


def _decode_task(task: str, logits: torch.Tensor) -> TaskPrediction:
    """Turn one task's logits into probabilities and a labelled prediction."""
    probabilities = torch.softmax(logits.reshape(-1), dim=-1)
    predicted_index = int(torch.argmax(probabilities).item())
    labels = TASK_CLASS_LABELS[task]

    if predicted_index >= len(labels):
        raise ValueError(
            f"Task '{task}' produced class index {predicted_index}, but only "
            f"{len(labels)} labels are defined."
        )

    return TaskPrediction(
        task=task,
        probabilities=[float(p) for p in probabilities],
        predicted_index=predicted_index,
        predicted_label=labels[predicted_index],
    )


def predict_batch(
    patients: list[Patient],
    artifacts: InferenceArtifacts,
) -> list[PredictionResult]:
    """Score several patients, one forward pass each.

    Args:
        patients: Validated patients carrying both active modalities.
        artifacts: Result of :func:`load_model`.

    Returns:
        One :class:`PredictionResult` per input patient, in input order.
    """
    return [predict(patient, artifacts) for patient in patients]
