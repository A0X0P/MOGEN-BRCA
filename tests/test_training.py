"""Loss, masking, and training-loop tests.

The masking contract (CLAUDE.md section 6) has two halves and both are pinned
here:

1.  A masked-out row must not change a task's loss *value* — losses are
    averaged over valid rows, and the Cox risk set contains only masked-in
    patients.
2.  A task with no valid rows in a batch must still yield a differentiable
    zero, so ``backward()`` succeeds on batches where a task is entirely
    absent.

The training smoke test then runs the real loop end to end on the synthetic
cohort: forward, all five losses, backward, optimizer step, validation,
checkpoint, and reload.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts import run_train
from src.data.datasets.multimodal_dataset import MultimodalDataset
from src.data.datasets.tabular_dataset import fit_normalization_stats
from src.data.pam50 import PAM50_GENES, PAM50_SUBTYPES
from src.data.schema.patient import Patient
from src.data.splits import split_patients
from src.data.tasks import ALL_TASKS, CLASSIFICATION_TASKS, IGNORE_INDEX
from src.inference.predict import load_model
from src.models.model_factory import build_model
from src.training.callbacks import CheckpointCallback, LoggingCallback
from src.training.losses import CoxLoss, FocalLoss, MultiTaskLoss, resolve_mask
from src.training.trainer import Trainer

SEED = 7


# --------------------------------------------------------------------------- #
# Mask resolution
# --------------------------------------------------------------------------- #
def test_ignore_index_alone_resolves_the_valid_rows() -> None:
    targets = torch.tensor([1, IGNORE_INDEX, 0, IGNORE_INDEX])

    assert resolve_mask(targets).tolist() == [True, False, True, False]


def test_explicit_mask_and_sentinel_are_combined() -> None:
    targets = torch.tensor([1, 0, IGNORE_INDEX, 0])
    mask = torch.tensor([True, False, True, True])

    assert resolve_mask(targets, mask).tolist() == [True, False, False, True]


# --------------------------------------------------------------------------- #
# Classification loss
# --------------------------------------------------------------------------- #
def test_masked_rows_do_not_change_the_loss_value() -> None:
    """Normalising over valid rows, not the batch, is what makes this hold."""
    loss_fn = FocalLoss()
    logits = torch.randn(6, 2, generator=torch.Generator().manual_seed(SEED))
    targets = torch.tensor([1, 0, 1, 0, 1, 0])
    mask = torch.tensor([True, False, True, False, False, True])

    masked = loss_fn(logits, targets, mask)
    subset = loss_fn(logits[mask], targets[mask])

    assert masked.item() == pytest.approx(subset.item(), rel=1e-6)


def test_changing_a_masked_out_label_cannot_move_the_loss() -> None:
    loss_fn = FocalLoss()
    logits = torch.randn(4, 2, generator=torch.Generator().manual_seed(SEED))
    mask = torch.tensor([True, True, False, False])

    first = loss_fn(logits, torch.tensor([1, 0, 1, 1]), mask)
    second = loss_fn(logits, torch.tensor([1, 0, 0, 0]), mask)

    assert first.item() == pytest.approx(second.item())


def test_sentinel_targets_are_excluded_without_an_explicit_mask() -> None:
    loss_fn = FocalLoss()
    logits = torch.randn(4, 5, generator=torch.Generator().manual_seed(SEED))
    targets = torch.tensor([2, IGNORE_INDEX, 4, IGNORE_INDEX])

    with_sentinel = loss_fn(logits, targets)
    explicit = loss_fn(logits[[0, 2]], torch.tensor([2, 4]))

    assert with_sentinel.item() == pytest.approx(explicit.item(), rel=1e-6)


def test_a_task_with_no_valid_rows_yields_a_differentiable_zero() -> None:
    loss_fn = FocalLoss()
    logits = torch.randn(3, 2, requires_grad=True)
    targets = torch.full((3,), IGNORE_INDEX)

    loss = loss_fn(logits, targets)
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad is True
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_focal_loss_rejects_a_negative_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        FocalLoss(gamma=-1.0)


# --------------------------------------------------------------------------- #
# Cox loss
# --------------------------------------------------------------------------- #
def test_cox_loss_matches_the_hand_computed_partial_likelihood() -> None:
    """Two patients, one event. -PL = log(1 + e) - 1."""
    loss = CoxLoss()(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 0.0]),
    )

    assert loss.item() == pytest.approx(math.log(1 + math.e) - 1.0, rel=1e-6)


def test_cox_loss_is_normalised_by_the_event_count() -> None:
    """With equal risks the loss reduces to mean(log |risk set|) over events."""
    n = 4
    loss = CoxLoss()(
        torch.zeros(n),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.ones(n),
    )

    expected = sum(math.log(size) for size in range(1, n + 1)) / n
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_cox_risk_set_excludes_masked_out_patients() -> None:
    """An excluded patient must not appear in anyone's risk denominator."""
    risk = torch.tensor([1.0, 0.0, 9.0])
    durations = torch.tensor([1.0, 2.0, 3.0])
    events = torch.tensor([1.0, 0.0, 0.0])
    mask = torch.tensor([True, True, False])

    masked = CoxLoss()(risk, durations, events, mask)
    subset = CoxLoss()(risk[:2], durations[:2], events[:2])

    assert masked.item() == pytest.approx(subset.item(), rel=1e-6)
    assert masked.item() == pytest.approx(math.log(1 + math.e) - 1.0, rel=1e-6)


def test_cox_loss_is_a_differentiable_zero_without_eligible_patients() -> None:
    risk = torch.randn(4, 1, requires_grad=True)

    loss = CoxLoss()(
        risk,
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([1.0, 1.0, 0.0, 1.0]),
        torch.zeros(4, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert risk.grad is not None
    assert torch.equal(risk.grad, torch.zeros_like(risk))


def test_cox_loss_is_a_differentiable_zero_without_observed_events() -> None:
    """The partial likelihood is undefined, not zero-risk: no fabricated event."""
    risk = torch.randn(3, requires_grad=True)

    loss = CoxLoss()(risk, torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3))
    loss.backward()

    assert loss.item() == 0.0
    assert risk.grad is not None


def test_cox_loss_ranks_a_correct_ordering_below_a_reversed_one() -> None:
    durations = torch.tensor([1.0, 2.0, 3.0])
    events = torch.tensor([1.0, 1.0, 1.0])

    correct = CoxLoss()(torch.tensor([2.0, 1.0, 0.0]), durations, events)
    reversed_order = CoxLoss()(torch.tensor([0.0, 1.0, 2.0]), durations, events)

    assert correct.item() < reversed_order.item()


# --------------------------------------------------------------------------- #
# Multi-task loss
# --------------------------------------------------------------------------- #
def _model_output(batch_size: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    return {
        "subtype_logits": torch.randn(
            batch_size, len(PAM50_SUBTYPES), generator=generator
        ),
        "er_logits": torch.randn(batch_size, 2, generator=generator),
        "pr_logits": torch.randn(batch_size, 2, generator=generator),
        "her2_logits": torch.randn(batch_size, 2, generator=generator),
        "risk_score": torch.randn(batch_size, 1, generator=generator),
    }


def _labels(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "subtype": torch.tensor([0, 1, 2, 3])[:batch_size],
        "er": torch.tensor([1, 0, 1, 0])[:batch_size],
        "pr": torch.tensor([1, 1, 0, 0])[:batch_size],
        "her2": torch.tensor([0, 0, 1, 0])[:batch_size],
        "duration": torch.tensor([12.0, 24.0, 36.0, 48.0])[:batch_size],
        "event": torch.tensor([1.0, 0.0, 1.0, 0.0])[:batch_size],
    }


def _all_valid(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {task: torch.ones(batch_size, dtype=torch.bool) for task in ALL_TASKS}


def test_multi_task_loss_reports_every_task_and_the_total() -> None:
    result = MultiTaskLoss()(_model_output(), _labels(), _all_valid())

    assert set(result) == {*ALL_TASKS, "total"}


def test_total_is_the_configured_weighted_sum() -> None:
    weights = {
        "subtype_weight": 2.0,
        "er_weight": 0.5,
        "pr_weight": 0.25,
        "her2_weight": 1.5,
        "survival_weight": 3.0,
    }
    result = MultiTaskLoss(**weights)(_model_output(), _labels(), _all_valid())

    expected = sum(
        weights[f"{task}_weight"] * result[task].item() for task in ALL_TASKS
    )
    assert result["total"].item() == pytest.approx(expected, rel=1e-6)


def test_a_batch_with_no_usable_labels_at_all_still_backpropagates() -> None:
    """The worst case a per-task mask can produce must not break training."""
    output = {key: tensor.requires_grad_(True) for key, tensor in _model_output().items()}
    labels = {**_labels()}
    for task in CLASSIFICATION_TASKS:
        labels[task] = torch.full((4,), IGNORE_INDEX)
    masks = {task: torch.zeros(4, dtype=torch.bool) for task in ALL_TASKS}

    result = MultiTaskLoss()(output, labels, masks)
    result["total"].backward()

    assert result["total"].item() == 0.0
    assert all(result[task].item() == 0.0 for task in ALL_TASKS)
    assert output["subtype_logits"].grad is not None
    assert output["risk_score"].grad is not None


def test_disabling_survival_drops_it_from_the_objective() -> None:
    result = MultiTaskLoss(enable_survival=False)(
        _model_output(), _labels(), _all_valid()
    )

    assert "survival" not in result


def test_a_missing_logit_key_raises_rather_than_being_skipped() -> None:
    output = _model_output()
    del output["her2_logits"]

    with pytest.raises(KeyError, match="her2_logits"):
        MultiTaskLoss()(output, _labels(), _all_valid())


def test_a_missing_label_raises() -> None:
    labels = _labels()
    del labels["pr"]

    with pytest.raises(KeyError, match="'pr'"):
        MultiTaskLoss()(_model_output(), labels, _all_valid())


def test_missing_risk_score_raises_when_survival_is_enabled() -> None:
    output = _model_output()
    del output["risk_score"]

    with pytest.raises(KeyError, match="risk_score"):
        MultiTaskLoss()(output, _labels(), _all_valid())


def test_missing_survival_labels_raise_when_survival_is_enabled() -> None:
    labels = _labels()
    del labels["duration"]

    with pytest.raises(KeyError, match="duration"):
        MultiTaskLoss()(_model_output(), labels, _all_valid())


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def _train_config(
    small_model_config: dict[str, Any],
    checkpoint_dir: Path,
    epochs: int = 2,
) -> dict[str, Any]:
    """A minimal config for a CPU smoke run."""
    return {
        "seed": SEED,
        "device": "cpu",
        "epochs": epochs,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "mixed_precision": False,
        "monitor": "val_total",
        "monitor_mode": "min",
        "scheduler": {"name": "plateau", "monitor": "val_total"},
        "checkpoint_dir": str(checkpoint_dir),
        "model": small_model_config,
    }


@pytest.fixture
def split_datasets(
    synthetic_cohort: list[Patient],
) -> tuple[dict[str, MultimodalDataset], dict[str, Any]]:
    """Train/val/test datasets built with train-fold statistics only."""
    split = split_patients(synthetic_cohort, seed=SEED)
    data_cfg: dict[str, Any] = {}
    run_train.fit_train_fold_statistics(split, data_cfg)
    return run_train.build_datasets(split, data_cfg), data_cfg


def test_statistics_are_fitted_on_the_train_fold_only(
    synthetic_cohort: list[Patient],
) -> None:
    """Fitting on the whole cohort would leak val/test into the features."""
    split = split_patients(synthetic_cohort, seed=SEED)
    data_cfg: dict[str, Any] = {}

    run_train.fit_train_fold_statistics(split, data_cfg)

    train_only = fit_normalization_stats(
        [p.clinical for p in split.train if p.clinical is not None]
    )
    whole_cohort = fit_normalization_stats(
        [p.clinical for p in synthetic_cohort if p.clinical is not None]
    )

    assert data_cfg["normalization_stats"] == train_only
    assert data_cfg["normalization_stats"] != whole_cohort
    assert data_cfg["gene_order"] == list(PAM50_GENES)
    assert len(data_cfg["gene_standardization"]["mean"]) == len(PAM50_GENES)


def test_training_smoke_run_completes_and_writes_a_checkpoint(
    split_datasets: tuple[dict[str, MultimodalDataset], dict[str, Any]],
    small_model_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Forward, all five losses, backward, step, validate, checkpoint."""
    datasets, data_cfg = split_datasets
    config = _train_config(small_model_config, tmp_path)
    config["data"] = data_cfg

    trainer = Trainer(
        model=build_model(small_model_config),
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        loss_fn=MultiTaskLoss(),
        config=config,
        callbacks=[
            CheckpointCallback(save_dir=tmp_path, monitor="val_total", mode="min"),
            LoggingCallback(),
        ],
    )
    history = trainer.train()

    for task in ALL_TASKS:
        assert f"train_{task}" in history
        assert f"val_{task}" in history
    assert len(history["train_total"]) == config["epochs"]
    assert all(math.isfinite(value) for value in history["train_total"])
    assert all(math.isfinite(value) for value in history["val_total"])
    assert (tmp_path / "checkpoint_best.pt").exists()


def test_validation_reports_masked_task_metrics(
    split_datasets: tuple[dict[str, MultimodalDataset], dict[str, Any]],
    small_model_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    datasets, data_cfg = split_datasets
    config = _train_config(small_model_config, tmp_path, epochs=1)
    config["data"] = data_cfg

    trainer = Trainer(
        model=build_model(small_model_config),
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        loss_fn=MultiTaskLoss(),
        config=config,
        callbacks=[],
    )
    history = trainer.train()

    assert "val_subtype_accuracy" in history
    assert "val_survival_concordance_index" in history
    assert 0.0 <= history["val_subtype_accuracy"][0] <= 1.0


def test_a_saved_checkpoint_reloads_into_an_identical_model(
    split_datasets: tuple[dict[str, MultimodalDataset], dict[str, Any]],
    small_model_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    datasets, data_cfg = split_datasets
    config = _train_config(small_model_config, tmp_path, epochs=1)
    config["data"] = data_cfg

    trainer = Trainer(
        model=build_model(small_model_config),
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        loss_fn=MultiTaskLoss(),
        config=config,
        callbacks=[
            CheckpointCallback(save_dir=tmp_path, monitor="val_total", mode="min")
        ],
    )
    trainer.train()

    artifacts = load_model(tmp_path / "checkpoint_best.pt")

    # The train-fold statistics travel with the checkpoint, so inference never
    # refits on its own cohort.
    assert artifacts.gene_order == PAM50_GENES
    assert artifacts.normalization_stats == data_cfg["normalization_stats"]
    assert artifacts.gene_standardization == data_cfg["gene_standardization"]

    sample = datasets["test"][0]
    inputs = {
        "clinical": sample["clinical"]["features"].unsqueeze(0),
        "genomics": sample["genomics"]["features"].unsqueeze(0),
    }
    trainer.model.eval()
    with torch.no_grad():
        expected = trainer.model(**inputs)["subtype_logits"]
        actual = artifacts.model(**inputs)["subtype_logits"]

    assert torch.allclose(expected, actual, atol=1e-6)


def test_training_resumes_from_a_checkpoint(
    split_datasets: tuple[dict[str, MultimodalDataset], dict[str, Any]],
    small_model_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    datasets, data_cfg = split_datasets
    config = _train_config(small_model_config, tmp_path, epochs=1)
    config["data"] = data_cfg

    def make_trainer(cfg: dict[str, Any]) -> Trainer:
        return Trainer(
            model=build_model(small_model_config),
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            loss_fn=MultiTaskLoss(),
            config=cfg,
            callbacks=[
                CheckpointCallback(save_dir=tmp_path, monitor="val_total", mode="min")
            ],
        )

    make_trainer(config).train()

    resumed_config = {
        **config,
        "epochs": 3,
        "resume_checkpoint": str(tmp_path / "checkpoint_best.pt"),
    }
    history = make_trainer(resumed_config).train()

    # Epoch 0 was already completed, so only epochs 1 and 2 run here.
    assert len(history["train_total"]) == 2


def test_an_unknown_scheduler_is_rejected(
    split_datasets: tuple[dict[str, MultimodalDataset], dict[str, Any]],
    small_model_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    datasets, _ = split_datasets
    config = _train_config(small_model_config, tmp_path)
    config["scheduler"] = {"name": "exponential-decay"}

    with pytest.raises(ValueError, match="Unknown scheduler"):
        Trainer(
            model=build_model(small_model_config),
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            loss_fn=MultiTaskLoss(),
            config=config,
        )
