"""Evaluation tests: mask-respecting metrics and the survival concordance index.

The rule under test is CLAUDE.md section 21: a task's metrics are computed over
its masked-in patients only, and a task with no usable labels in a partition
reports *no metrics* rather than a fabricated score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.datasets.multimodal_dataset import MultimodalDataset
from src.data.schema.patient import Patient
from src.data.tasks import ALL_TASKS, CLASSIFICATION_TASKS, RECEPTOR_TASKS
from src.evaluation import metrics
from src.evaluation.evaluator import (
    EvaluationResult,
    PredictionCollector,
    evaluate,
    extract_labels,
    extract_masks,
)
from src.inference.predict import load_model, predict
from src.models.model_factory import MultimodalCancerModel, build_model
from src.training.trainer import collate_multimodal
from src.utils.io import save_checkpoint
from tests.conftest import build_cohort

BATCH_SIZE = 16


@pytest.fixture
def model(small_model_config: dict[str, Any]) -> MultimodalCancerModel:
    """A narrow model at the ratified contract dimensions, in eval mode."""
    torch.manual_seed(0)
    built = build_model(small_model_config)
    built.eval()
    return built


def _loader(dataset: MultimodalDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_multimodal,
    )


# --------------------------------------------------------------------------- #
# Mask-respecting evaluation
# --------------------------------------------------------------------------- #
def test_evaluation_counts_match_the_dataset_masks(
    synthetic_dataset: MultimodalDataset,
    model: MultimodalCancerModel,
) -> None:
    result = evaluate(model, _loader(synthetic_dataset))

    assert result.n_samples == len(synthetic_dataset)
    assert result.task_counts == synthetic_dataset.mask_counts()


def test_metrics_are_reported_for_every_task_with_usable_labels(
    synthetic_dataset: MultimodalDataset,
    model: MultimodalCancerModel,
) -> None:
    result = evaluate(model, _loader(synthetic_dataset))

    for task in CLASSIFICATION_TASKS:
        assert set(result.classification[task]) >= {
            "accuracy",
            "precision",
            "recall",
            "f1",
        }
        assert 0.0 <= result.classification[task]["accuracy"] <= 1.0

    for task in RECEPTOR_TASKS:
        assert 0.0 <= result.calibration[task]["brier_score"] <= 1.0

    assert result.survival is not None
    assert 0.0 <= result.survival["concordance_index"] <= 1.0


def test_a_task_with_no_usable_labels_reports_no_metrics(
    train_statistics: dict[str, Any],
    model: MultimodalCancerModel,
) -> None:
    """No labels must mean no score — not a substituted or default value."""
    cohort = build_cohort()
    for patient in cohort:
        patient.targets.her2_positive = None

    dataset = MultimodalDataset(
        patients=cohort,
        normalization_stats=train_statistics["normalization_stats"],
        gene_standardization=train_statistics["gene_standardization"],
    )
    result = evaluate(model, _loader(dataset))

    assert result.task_counts["her2"] == 0
    assert "her2" not in result.classification
    assert "her2" not in result.calibration
    assert "her2" not in result.flat_metrics()
    # The other tasks are unaffected.
    assert result.classification["er"]


def test_masked_out_rows_cannot_influence_a_task_metric(
    synthetic_dataset: MultimodalDataset,
    model: MultimodalCancerModel,
) -> None:
    """Changing a masked-out label must leave that task's metrics identical."""
    batch = collate_multimodal([synthetic_dataset[i] for i in range(BATCH_SIZE)])
    device = torch.device("cpu")
    labels = extract_labels(batch, device)
    masks = extract_masks(batch, device)

    with torch.no_grad():
        output = model(
            clinical=batch["clinical"]["features"],
            genomics=batch["genomics"]["features"],
        )

    baseline = PredictionCollector()
    baseline.update(output, labels, masks)

    tampered_labels = {key: value.clone() for key, value in labels.items()}
    for task in CLASSIFICATION_TASKS:
        tampered_labels[task][~masks[task]] = 1
    tampered_labels["duration"][~masks["survival"]] = 999.0
    tampered_labels["event"][~masks["survival"]] = 1.0

    tampered = PredictionCollector()
    tampered.update(output, tampered_labels, masks)

    assert tampered.result().classification == baseline.result().classification
    assert tampered.result().survival == baseline.result().survival


def test_all_masked_survival_reports_no_survival_metrics(
    train_statistics: dict[str, Any],
    model: MultimodalCancerModel,
) -> None:
    cohort = build_cohort()
    for patient in cohort:
        patient.targets.os_months = None
        patient.targets.os_event = None
        patient.targets.survival_excluded = False
        patient.targets.survival_exclusion_reason = None

    dataset = MultimodalDataset(
        patients=cohort,
        normalization_stats=train_statistics["normalization_stats"],
        gene_standardization=train_statistics["gene_standardization"],
    )
    result = evaluate(model, _loader(dataset))

    assert result.survival is None
    assert result.task_counts["survival"] == 0


def test_evaluation_leaves_the_model_weights_untouched(
    synthetic_dataset: MultimodalDataset,
    model: MultimodalCancerModel,
) -> None:
    before = {name: p.clone() for name, p in model.named_parameters()}

    evaluate(model, _loader(synthetic_dataset))

    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
    assert model.training is False


# --------------------------------------------------------------------------- #
# Result shaping
# --------------------------------------------------------------------------- #
def test_flat_metrics_prefixes_every_value_with_its_task() -> None:
    result = EvaluationResult(
        classification={"subtype": {"accuracy": 0.5}},
        calibration={"er": {"brier_score": 0.2}},
        survival={"concordance_index": 0.7},
        task_counts={"subtype": 10},
        n_samples=10,
    )

    assert result.flat_metrics() == {
        "subtype_accuracy": 0.5,
        "er_brier_score": 0.2,
        "survival_concordance_index": 0.7,
    }


def test_to_dict_is_json_serialisable(
    synthetic_dataset: MultimodalDataset,
    model: MultimodalCancerModel,
) -> None:
    import json

    payload = evaluate(model, _loader(synthetic_dataset)).to_dict()

    assert set(payload) == {
        "classification",
        "survival",
        "calibration",
        "task_counts",
        "n_samples",
    }
    json.dumps(payload)  # raises if a non-serialisable value crept in


def test_collector_uses_all_rows_when_no_mask_is_supplied(
    synthetic_dataset: MultimodalDataset,
    model: MultimodalCancerModel,
) -> None:
    """Absent masks mean "nothing to exclude", not "exclude everything"."""
    batch = collate_multimodal([synthetic_dataset[i] for i in range(BATCH_SIZE)])
    device = torch.device("cpu")
    labels = extract_labels(batch, device)
    row = torch.arange(BATCH_SIZE)
    labels = {
        "subtype": row % 5,
        "er": row % 2,
        "pr": (row + 1) % 2,
        "her2": row % 2,
        "duration": labels["duration"].clamp(min=1.0),
        "event": torch.ones_like(labels["event"]),
    }

    collector = PredictionCollector()
    with torch.no_grad():
        output = model(
            clinical=batch["clinical"]["features"],
            genomics=batch["genomics"]["features"],
        )
    collector.update(output, labels, {})
    result = collector.result()

    assert result.task_counts["subtype"] == BATCH_SIZE
    assert result.task_counts["survival"] == BATCH_SIZE


def test_partial_masks_select_exactly_their_rows(
    synthetic_dataset: MultimodalDataset,
    model: MultimodalCancerModel,
) -> None:
    batch = collate_multimodal([synthetic_dataset[i] for i in range(BATCH_SIZE)])
    device = torch.device("cpu")
    masks = extract_masks(batch, device)
    masks = {task: mask.clone() for task, mask in masks.items()}
    masks["er"] = torch.zeros(BATCH_SIZE, dtype=torch.bool)
    masks["er"][:3] = True

    collector = PredictionCollector()
    with torch.no_grad():
        output = model(
            clinical=batch["clinical"]["features"],
            genomics=batch["genomics"]["features"],
        )
    collector.update(output, extract_labels(batch, device), masks)

    assert collector.result().task_counts["er"] == 3


# --------------------------------------------------------------------------- #
# Concordance index
# --------------------------------------------------------------------------- #
def test_concordance_index_is_one_for_a_perfect_ranking() -> None:
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([1.0, 1.0, 1.0, 1.0])
    risk = np.array([4.0, 3.0, 2.0, 1.0])

    assert metrics.concordance_index(risk, durations, events) == pytest.approx(1.0)


def test_concordance_index_is_zero_for_a_reversed_ranking() -> None:
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([1.0, 1.0, 1.0, 1.0])
    risk = np.array([1.0, 2.0, 3.0, 4.0])

    assert metrics.concordance_index(risk, durations, events) == pytest.approx(0.0)


def test_tied_risks_score_one_half() -> None:
    durations = np.array([1.0, 2.0])
    events = np.array([1.0, 1.0])

    assert metrics.concordance_index(
        np.array([0.5, 0.5]), durations, events
    ) == pytest.approx(0.5)


def test_concordance_index_without_comparable_pairs_is_one_half() -> None:
    """All-censored data yields no comparable pair; 0.5 is reported, not an error."""
    durations = np.array([1.0, 2.0, 3.0])
    events = np.array([0.0, 0.0, 0.0])

    assert metrics.concordance_index(
        np.array([3.0, 2.0, 1.0]), durations, events
    ) == pytest.approx(0.5)


def test_censored_patients_only_enter_as_comparison_partners() -> None:
    """A censored patient is never the earlier member of a comparable pair."""
    risk = np.array([2.0, 1.0])
    durations = np.array([1.0, 5.0])

    event_first = metrics.concordance_index(risk, durations, np.array([1.0, 0.0]))
    censored_first = metrics.concordance_index(risk, durations, np.array([0.0, 1.0]))

    assert event_first == pytest.approx(1.0)
    assert censored_first == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Checkpoint round trip
# --------------------------------------------------------------------------- #
def test_reloaded_checkpoint_reproduces_the_evaluation(
    synthetic_dataset: MultimodalDataset,
    model: MultimodalCancerModel,
    small_model_config: dict[str, Any],
    train_statistics: dict[str, Any],
    tmp_path: Path,
) -> None:
    baseline = evaluate(model, _loader(synthetic_dataset))

    checkpoint_path = tmp_path / "checkpoint_best.pt"
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "config": {
                "device": "cpu",
                "model": small_model_config,
                "data": {
                    "normalization_stats": train_statistics["normalization_stats"],
                    "gene_standardization": train_statistics["gene_standardization"],
                },
            },
        },
        checkpoint_path,
    )

    artifacts = load_model(checkpoint_path)
    reloaded = evaluate(artifacts.model, _loader(synthetic_dataset))

    assert reloaded.task_counts == baseline.task_counts
    assert reloaded.classification == baseline.classification
    assert reloaded.survival == baseline.survival


def test_single_patient_inference_agrees_with_the_batched_forward(
    synthetic_cohort: list[Patient],
    model: MultimodalCancerModel,
    small_model_config: dict[str, Any],
    train_statistics: dict[str, Any],
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint_best.pt"
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "config": {
                "device": "cpu",
                "model": small_model_config,
                "data": {
                    "normalization_stats": train_statistics["normalization_stats"],
                    "gene_standardization": train_statistics["gene_standardization"],
                },
            },
        },
        checkpoint_path,
    )
    artifacts = load_model(checkpoint_path)
    patient = synthetic_cohort[0]

    result = predict(patient, artifacts)

    assert result.patient_id == patient.patient_id
    assert set(result.tasks) == set(CLASSIFICATION_TASKS)
    assert result.risk_score is not None
    assert sum(result.tasks["subtype"].probabilities) == pytest.approx(1.0, abs=1e-5)
    assert result.modalities_used == ["clinical", "genomics"]


def test_evaluation_task_counts_cover_every_active_task(
    synthetic_dataset: MultimodalDataset,
    model: MultimodalCancerModel,
) -> None:
    result = evaluate(model, _loader(synthetic_dataset))

    assert set(result.task_counts) == set(ALL_TASKS)


def test_classification_metric_helpers_agree_with_hand_counts() -> None:
    """A sanity anchor for the metric wrappers themselves."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    assert metrics.accuracy(y_true, y_pred) == pytest.approx(0.75)
    assert metrics.recall(y_true, y_pred, average="binary") == pytest.approx(1.0)
    assert metrics.precision(y_true, y_pred, average="binary") == pytest.approx(2 / 3)
