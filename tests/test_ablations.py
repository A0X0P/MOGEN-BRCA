"""Tests for the single-modality ablations and the SHAP explainability path.

The ablations must be a *configuration* of the ratified architecture, not a
second architecture. These tests pin that:

- each ablation config differs from the full model config only in which
  modality is enabled;
- each ablation training config copies the frozen run's optimisation protocol
  verbatim, so a measured difference is attributable to the modality rather
  than to a retuned learning rate or epoch budget;
- an ablation model keeps all five task heads and drops the cross-modal
  attention entirely, rather than instantiating it and leaving it inert;
- the evaluator and trainer route only the modalities a model actually accepts.

The SHAP tests pin the joint 62-feature input space used to explain the frozen
multimodal checkpoint: the wrapper must reproduce the model's own forward pass
exactly, and the feature names must be the real gene symbols and the real
clinical feature names, in the real input order.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from scripts.run_shap import (
    CLINICAL_MODALITY,
    EXPLAINED_OUTPUTS,
    GENOMIC_MODALITY,
    JointInputModel,
    build_feature_matrix,
    feature_names,
    select_background,
)
from src.data.datasets.multimodal_dataset import MultimodalDataset
from src.data.pam50 import PAM50_GENE_COUNT, PAM50_GENES, PAM50_SUBTYPES
from src.data.schema.clinical import CLINICAL_FEATURE_DIM, CLINICAL_FEATURE_NAMES
from src.data.schema.patient import Patient
from src.data.tasks import RECEPTOR_TASKS, RISK_SCORE_KEY, TASK_LOGIT_KEYS
from src.evaluation.evaluator import ACTIVE_MODALITIES, build_model_inputs
from src.inference.predict import InferenceArtifacts
from src.models.model_factory import MultimodalCancerModel, build_model
from src.training.losses import MultiTaskLoss
from src.training.trainer import collate_multimodal
from src.utils.io import load_yaml
from tests.conftest import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "configs" / "breast"

FULL_MODEL_CONFIG = CONFIG_DIR / "model.yaml"
GENOMICS_ONLY_MODEL_CONFIG = CONFIG_DIR / "model_genomics_only.yaml"
CLINICAL_ONLY_MODEL_CONFIG = CONFIG_DIR / "model_clinical_only.yaml"

FULL_TRAIN_CONFIG = CONFIG_DIR / "train.yaml"
GENOMICS_ONLY_TRAIN_CONFIG = CONFIG_DIR / "train_genomics_only.yaml"
CLINICAL_ONLY_TRAIN_CONFIG = CONFIG_DIR / "train_clinical_only.yaml"

#: Training keys that must be byte-identical across the three runs, so that a
#: metric difference cannot be explained by a different optimisation protocol.
SHARED_TRAINING_KEYS = (
    "data_config",
    "seed",
    "device",
    "mixed_precision",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "grad_accumulation_steps",
    "grad_clip",
    "scheduler",
    "loss",
    "monitor",
    "monitor_mode",
    "early_stopping_patience",
)

BATCH_SIZE = 4


@pytest.fixture
def genomics_only_config(small_model_config: dict[str, Any]) -> dict[str, Any]:
    """The narrow test config with the clinical branch disabled."""
    return {**small_model_config, "enable_clinical": False}


@pytest.fixture
def clinical_only_config(small_model_config: dict[str, Any]) -> dict[str, Any]:
    """The narrow test config with the genomic branch disabled."""
    return {**small_model_config, "enable_genomics": False}


def _batch(dataset: MultimodalDataset, size: int = BATCH_SIZE) -> dict[str, Any]:
    """Collate the first ``size`` patients into a training batch."""
    return collate_multimodal([dataset[index] for index in range(size)])


# --------------------------------------------------------------------------- #
# Checked-in ablation configs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("path", "disabled"),
    [
        (GENOMICS_ONLY_MODEL_CONFIG, "enable_clinical"),
        (CLINICAL_ONLY_MODEL_CONFIG, "enable_genomics"),
    ],
)
def test_ablation_model_config_differs_only_in_the_disabled_modality(
    path: Any, disabled: str
) -> None:
    """An ablation must not quietly change capacity or head dimensions."""
    full = dict(load_yaml(FULL_MODEL_CONFIG))
    ablation = dict(load_yaml(path))

    assert set(ablation) == set(full)
    assert ablation[disabled] is False
    assert full[disabled] is True

    differing = {key for key in full if full[key] != ablation[key]}
    assert differing == {disabled}


@pytest.mark.parametrize(
    ("path", "expected_output_dir"),
    [
        (GENOMICS_ONLY_TRAIN_CONFIG, "results/breast_ablation/genomics_only"),
        (CLINICAL_ONLY_TRAIN_CONFIG, "results/breast_ablation/clinical_only"),
    ],
)
def test_ablation_train_config_copies_the_frozen_protocol(
    path: Any, expected_output_dir: str
) -> None:
    """Same optimisation protocol, same split inputs, a separate output tree."""
    full = dict(load_yaml(FULL_TRAIN_CONFIG))
    ablation = dict(load_yaml(path))

    for key in SHARED_TRAINING_KEYS:
        assert ablation[key] == full[key], f"'{key}' diverges from the frozen run"

    assert ablation["seed"] == 42
    assert ablation["output_dir"] == expected_output_dir
    assert ablation["output_dir"] != full["output_dir"]
    assert not ablation["checkpoint_dir"].startswith("results/breast/")


def test_ablation_configs_never_write_into_the_frozen_run_directory() -> None:
    """The frozen results tree must be unreachable from an ablation config."""
    for path in (GENOMICS_ONLY_TRAIN_CONFIG, CLINICAL_ONLY_TRAIN_CONFIG):
        config = dict(load_yaml(path))
        for key in ("output_dir", "checkpoint_dir"):
            assert config[key].startswith("results/breast_ablation/")


# --------------------------------------------------------------------------- #
# Ablation model structure
# --------------------------------------------------------------------------- #
def test_genomics_only_model_drops_the_clinical_branch_and_cross_attention(
    genomics_only_config: dict[str, Any],
) -> None:
    built = build_model(genomics_only_config)

    assert built.clinical_encoder is None
    assert built.genomics_encoder is not None
    assert built.cross_attention is None
    assert "cross_attention" not in dict(built.named_children())


def test_clinical_only_model_drops_the_genomic_branch_and_cross_attention(
    clinical_only_config: dict[str, Any],
) -> None:
    built = build_model(clinical_only_config)

    assert built.genomics_encoder is None
    assert built.clinical_encoder is not None
    assert built.cross_attention is None
    assert "cross_attention" not in dict(built.named_children())


def test_full_model_keeps_cross_attention(
    small_model_config: dict[str, Any],
) -> None:
    """Cross-modal attention is only omitted for a single-modality ablation."""
    built = build_model(small_model_config)

    assert built.cross_attention is not None


@pytest.mark.parametrize(
    ("config_key", "expected"),
    [
        ("enable_clinical", ("genomics",)),
        ("enable_genomics", ("clinical",)),
    ],
)
def test_active_modalities_reports_only_the_enabled_modality(
    small_model_config: dict[str, Any], config_key: str, expected: tuple[str, ...]
) -> None:
    built = build_model({**small_model_config, config_key: False})

    assert built.active_modalities == expected


def test_full_model_active_modalities_is_the_canonical_pair(
    small_model_config: dict[str, Any],
) -> None:
    built = build_model(small_model_config)

    assert built.active_modalities == ACTIVE_MODALITIES == ("clinical", "genomics")


# --------------------------------------------------------------------------- #
# Ablation forward and backward
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("config_key", "modality", "width"),
    [
        ("enable_clinical", "genomics", PAM50_GENE_COUNT),
        ("enable_genomics", "clinical", CLINICAL_FEATURE_DIM),
    ],
)
def test_ablation_model_keeps_all_five_task_heads(
    small_model_config: dict[str, Any], config_key: str, modality: str, width: int
) -> None:
    """Dropping a modality must not drop a task."""
    built = build_model({**small_model_config, config_key: False}).eval()

    with torch.no_grad():
        output = built(**{modality: torch.randn(BATCH_SIZE, width)})

    assert output[TASK_LOGIT_KEYS["subtype"]].shape == (
        BATCH_SIZE,
        len(PAM50_SUBTYPES),
    )
    for task in RECEPTOR_TASKS:
        assert output[TASK_LOGIT_KEYS[task]].shape == (BATCH_SIZE, 2)
    assert output[RISK_SCORE_KEY].shape[0] == BATCH_SIZE
    assert all(torch.isfinite(value).all() for value in output.values())


@pytest.mark.parametrize("config_key", ["enable_clinical", "enable_genomics"])
def test_ablation_backward_reaches_every_parameter(
    small_model_config: dict[str, Any],
    synthetic_dataset: MultimodalDataset,
    config_key: str,
) -> None:
    """No parameter may be left stranded by the removed branch."""
    built = build_model({**small_model_config, config_key: False})
    batch = _batch(synthetic_dataset)

    inputs = build_model_inputs(batch, torch.device("cpu"), built.active_modalities)
    output = built(**inputs)

    labels = {task: batch["label"][task] for task in ("subtype", *RECEPTOR_TASKS)}
    labels["duration"] = batch["survival"]["duration"]
    labels["event"] = batch["survival"]["event"]
    MultiTaskLoss()(output, labels, batch["mask"])["total"].backward()

    stranded = [
        name
        for name, parameter in built.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
    ]
    assert stranded == []


@pytest.mark.parametrize(
    ("config_key", "supplied", "message"),
    [
        ("enable_genomics", "genomics", "genomic encoder is disabled"),
        ("enable_clinical", "clinical", "clinical encoder is disabled"),
    ],
)
def test_supplying_a_disabled_modality_raises(
    small_model_config: dict[str, Any], config_key: str, supplied: str, message: str
) -> None:
    """A disabled modality's tensor must fail loudly, never be ignored."""
    built = build_model({**small_model_config, config_key: False})
    width = PAM50_GENE_COUNT if supplied == "genomics" else CLINICAL_FEATURE_DIM

    with pytest.raises(ValueError, match=message):
        built(**{supplied: torch.randn(BATCH_SIZE, width)})


# --------------------------------------------------------------------------- #
# Modality routing
# --------------------------------------------------------------------------- #
def test_build_model_inputs_defaults_to_both_modalities(
    synthetic_dataset: MultimodalDataset,
) -> None:
    inputs = build_model_inputs(_batch(synthetic_dataset), torch.device("cpu"))

    assert set(inputs) == set(ACTIVE_MODALITIES)


@pytest.mark.parametrize("modality", ["clinical", "genomics"])
def test_build_model_inputs_filters_to_the_requested_modality(
    synthetic_dataset: MultimodalDataset, modality: str
) -> None:
    batch = _batch(synthetic_dataset)

    inputs = build_model_inputs(batch, torch.device("cpu"), (modality,))

    assert set(inputs) == {modality}
    assert torch.equal(inputs[modality], batch[modality]["features"])


def test_build_model_inputs_rejects_an_unknown_modality(
    synthetic_dataset: MultimodalDataset,
) -> None:
    with pytest.raises(ValueError, match="Unsupported modalities"):
        build_model_inputs(
            _batch(synthetic_dataset), torch.device("cpu"), ("imaging",)
        )


def test_build_model_inputs_rejects_an_empty_modality_list(
    synthetic_dataset: MultimodalDataset,
) -> None:
    with pytest.raises(ValueError, match="At least one modality"):
        build_model_inputs(_batch(synthetic_dataset), torch.device("cpu"), ())


def test_build_model_inputs_reports_a_missing_modality(
    synthetic_dataset: MultimodalDataset,
) -> None:
    batch = _batch(synthetic_dataset)
    del batch["genomics"]

    with pytest.raises(KeyError, match="missing the 'genomics' modality"):
        build_model_inputs(batch, torch.device("cpu"), ("genomics",))


# --------------------------------------------------------------------------- #
# SHAP joint input space
# --------------------------------------------------------------------------- #
@pytest.fixture
def artifacts(
    small_model_config: dict[str, Any], train_statistics: dict[str, Any]
) -> InferenceArtifacts:
    """A built model paired with the preprocessing a checkpoint would carry."""
    built = build_model(small_model_config)
    built.eval()
    return InferenceArtifacts(
        model=built,
        config={"model": small_model_config},
        device=torch.device("cpu"),
        normalization_stats=train_statistics["normalization_stats"],
        gene_standardization=train_statistics["gene_standardization"],
        gene_order=PAM50_GENES,
    )


def test_feature_names_are_the_real_gene_and_clinical_names(
    artifacts: InferenceArtifacts,
) -> None:
    """No fabricated feature names, and the two blocks stay separable."""
    names, modalities = feature_names(artifacts)

    assert len(names) == PAM50_GENE_COUNT + CLINICAL_FEATURE_DIM == 62
    assert names[:PAM50_GENE_COUNT] == tuple(PAM50_GENES)
    assert names[PAM50_GENE_COUNT:] == tuple(CLINICAL_FEATURE_NAMES)
    assert set(modalities[:PAM50_GENE_COUNT]) == {GENOMIC_MODALITY}
    assert set(modalities[PAM50_GENE_COUNT:]) == {CLINICAL_MODALITY}


def test_feature_matrix_puts_genomics_first_and_matches_the_dataset(
    artifacts: InferenceArtifacts,
    synthetic_cohort: list[Patient],
    synthetic_dataset: MultimodalDataset,
) -> None:
    """The joint matrix must be the dataset's own encoding, just concatenated."""
    patients = synthetic_cohort[:BATCH_SIZE]

    matrix, patient_ids = build_feature_matrix(patients, artifacts)

    assert matrix.shape == (BATCH_SIZE, 62)
    assert patient_ids == [patient.patient_id for patient in patients]
    for index in range(BATCH_SIZE):
        row = synthetic_dataset[index]
        assert torch.equal(matrix[index, :PAM50_GENE_COUNT], row["genomics"]["features"])
        assert torch.equal(matrix[index, PAM50_GENE_COUNT:], row["clinical"]["features"])


@pytest.mark.parametrize("output", EXPLAINED_OUTPUTS, ids=lambda o: o.task)
def test_joint_wrapper_reproduces_the_direct_model_call(
    artifacts: InferenceArtifacts, synthetic_cohort: list[Patient], output: Any
) -> None:
    """The explained function must be the model's own forward pass, exactly."""
    matrix, _ = build_feature_matrix(synthetic_cohort[:BATCH_SIZE], artifacts)
    wrapper = JointInputModel(artifacts.model, output, PAM50_GENE_COUNT).eval()

    with torch.no_grad():
        through_wrapper = wrapper(matrix)
        direct = output.select(
            artifacts.model(
                clinical=matrix[:, PAM50_GENE_COUNT:],
                genomics=matrix[:, :PAM50_GENE_COUNT],
            )
        )

    assert torch.equal(through_wrapper, direct)
    assert through_wrapper.shape == (BATCH_SIZE, len(output.output_names))


def test_joint_wrapper_rejects_a_non_matrix_input(
    artifacts: InferenceArtifacts,
) -> None:
    wrapper = JointInputModel(artifacts.model, EXPLAINED_OUTPUTS[0], PAM50_GENE_COUNT)

    with pytest.raises(ValueError, match="2-D"):
        wrapper(torch.randn(62))


def test_explained_outputs_cover_all_five_tasks() -> None:
    """Every task in the architecture must have a documented explained output."""
    tasks = {output.task for output in EXPLAINED_OUTPUTS}

    assert tasks == {"er", "pr", "her2", "pam50", "survival"}

    pam50 = next(o for o in EXPLAINED_OUTPUTS if o.task == "pam50")
    assert pam50.output_names == PAM50_SUBTYPES
    for output in EXPLAINED_OUTPUTS:
        assert output.title, f"{output.task} must document what is explained"


def test_background_selection_is_deterministic_and_bounded(
    synthetic_cohort: list[Patient],
) -> None:
    """The SHAP background must be reproducible from the seed alone."""
    first = select_background(synthetic_cohort, 10, seed=42)
    second = select_background(synthetic_cohort, 10, seed=42)
    other = select_background(synthetic_cohort, 10, seed=7)

    assert [p.patient_id for p in first] == [p.patient_id for p in second]
    assert len(first) == 10
    assert [p.patient_id for p in first] != [p.patient_id for p in other]

    everything = select_background(synthetic_cohort, len(synthetic_cohort) + 5, seed=42)
    assert len(everything) == len(synthetic_cohort)
