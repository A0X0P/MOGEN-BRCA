"""Model construction and forward-pass tests for the two-modality model.

The architecture under test is the ratified one:

    Gene expression (50)          Clinical (12)
            |                           |
    Genomic Transformer            Clinical MLP
            \\                         /
             Cross-modal attention
                       |
                 Concatenation
                       |
                   Fusion MLP
                       |
      PAM50 / ER / PR / HER2 / DeepSurv heads

These tests pin the contract dimensions, the five head outputs, the absence of
any imaging branch, and that a real batch from the dataset flows through the
model unchanged in shape.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.data.datasets.multimodal_dataset import MultimodalDataset
from src.data.pam50 import PAM50_GENE_COUNT, PAM50_SUBTYPES
from src.data.schema.clinical import CLINICAL_FEATURE_DIM
from src.data.tasks import RECEPTOR_TASKS, RISK_SCORE_KEY, TASK_LOGIT_KEYS
from src.evaluation.evaluator import build_model_inputs
from src.models.classification.multitask_head import MultiTaskClassificationHead
from src.models.fusion.cross_modal_attention import CrossModalAttention
from src.models.fusion.fusion_head import FusionHead
from src.models.genomics.genomic_transformer import GenomicTransformer
from src.models.model_factory import ModelConfig, MultimodalCancerModel, build_model
from src.models.survival.deepsurv_head import DeepSurvHead
from src.models.tabular.clinical_mlp import ClinicalMLP
from src.training.trainer import collate_multimodal
from src.utils.io import load_yaml
from tests.conftest import REPO_ROOT

MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "breast" / "model.yaml"

BATCH_SIZE = 4


@pytest.fixture
def model(small_model_config: dict[str, Any]) -> MultimodalCancerModel:
    """A narrow model at the ratified contract dimensions, in eval mode."""
    built = build_model(small_model_config)
    built.eval()
    return built


def _inputs(batch_size: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    """Random inputs shaped to the two-modality contract."""
    torch.manual_seed(0)
    return {
        "clinical": torch.randn(batch_size, CLINICAL_FEATURE_DIM),
        "genomics": torch.randn(batch_size, PAM50_GENE_COUNT),
    }


# --------------------------------------------------------------------------- #
# Configuration contract
# --------------------------------------------------------------------------- #
def test_default_config_matches_the_ratified_contracts() -> None:
    """The defaults, not just the YAML, encode the contract."""
    config = ModelConfig()

    assert config.clinical_input_dim == CLINICAL_FEATURE_DIM == 12
    assert config.genomics_input_dim == PAM50_GENE_COUNT == 50
    assert config.subtype_num_classes == len(PAM50_SUBTYPES) == 5
    assert config.receptor_num_classes == 2
    assert config.enable_clinical is True
    assert config.enable_genomics is True
    assert config.enable_survival is True


def test_repository_model_config_builds_at_the_contract_dimensions() -> None:
    """The checked-in config must agree with the contract, not just parse."""
    config = dict(load_yaml(MODEL_CONFIG_PATH))

    assert config["clinical_input_dim"] == CLINICAL_FEATURE_DIM
    assert config["genomics_input_dim"] == PAM50_GENE_COUNT
    assert config["subtype_num_classes"] == len(PAM50_SUBTYPES)
    assert config["receptor_num_classes"] == 2

    built = build_model(config)

    assert built.clinical_encoder is not None
    assert built.genomics_encoder is not None
    assert built.survival_head is not None
    assert built.genomics_encoder.input_dim == PAM50_GENE_COUNT


def test_build_model_ignores_keys_that_are_not_model_fields(
    small_model_config: dict[str, Any],
) -> None:
    """Data/training sections may be passed through without breaking the build."""
    built = build_model({**small_model_config, "learning_rate": 1e-4, "epochs": 3})

    assert isinstance(built, MultimodalCancerModel)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_model_contains_exactly_the_ratified_components(
    model: MultimodalCancerModel,
) -> None:
    children = dict(model.named_children())

    assert set(children) == {
        "clinical_encoder",
        "genomics_encoder",
        "cross_attention",
        "fusion_head",
        "classification_head",
        "survival_head",
    }
    assert isinstance(children["clinical_encoder"], ClinicalMLP)
    assert isinstance(children["genomics_encoder"], GenomicTransformer)
    assert isinstance(children["cross_attention"], CrossModalAttention)
    assert isinstance(children["fusion_head"], FusionHead)
    assert isinstance(children["classification_head"], MultiTaskClassificationHead)
    assert isinstance(children["survival_head"], DeepSurvHead)


def test_model_has_no_imaging_branch(model: MultimodalCancerModel) -> None:
    """Imaging is out of scope: no module or config field may reintroduce it."""
    module_names = {name.lower() for name, _ in model.named_modules()}
    imaging_terms = ("imag", "encoder_2d", "encoder_3d", "cnn", "efficientnet", "dicom")

    assert not [
        name for name in module_names if any(term in name for term in imaging_terms)
    ]
    assert not [
        field
        for field in vars(ModelConfig())
        if any(term in field.lower() for term in imaging_terms)
    ]


def test_model_parameters_are_all_trainable(model: MultimodalCancerModel) -> None:
    assert sum(p.numel() for p in model.parameters()) > 0
    assert all(p.requires_grad for p in model.parameters())


# --------------------------------------------------------------------------- #
# Forward pass
# --------------------------------------------------------------------------- #
def test_forward_returns_all_five_task_outputs(model: MultimodalCancerModel) -> None:
    output = model(**_inputs())

    assert set(output) == {
        *TASK_LOGIT_KEYS.values(),
        "fused",
        RISK_SCORE_KEY,
    }


def test_forward_head_shapes_follow_the_task_contracts(
    model: MultimodalCancerModel,
) -> None:
    output = model(**_inputs())

    assert output["subtype_logits"].shape == (BATCH_SIZE, len(PAM50_SUBTYPES))
    for task in RECEPTOR_TASKS:
        assert output[TASK_LOGIT_KEYS[task]].shape == (BATCH_SIZE, 2)
    assert output[RISK_SCORE_KEY].shape == (BATCH_SIZE, 1)
    assert output["fused"].shape == (BATCH_SIZE, model.config.fused_dim)


def test_forward_outputs_are_finite(model: MultimodalCancerModel) -> None:
    output = model(**_inputs())

    assert all(torch.isfinite(tensor).all() for tensor in output.values())


@pytest.mark.parametrize("batch_size", [1, 2, 7])
def test_forward_handles_any_batch_size(
    batch_size: int, model: MultimodalCancerModel
) -> None:
    output = model(**_inputs(batch_size))

    assert output["subtype_logits"].shape[0] == batch_size
    assert output[RISK_SCORE_KEY].shape[0] == batch_size


def test_every_head_is_differentiable_through_both_encoders(
    small_model_config: dict[str, Any],
) -> None:
    built = build_model(small_model_config)
    output = built(**_inputs())

    total = sum(
        output[key].sum() for key in (*TASK_LOGIT_KEYS.values(), RISK_SCORE_KEY)
    )
    total.backward()

    encoders = ("clinical_encoder", "genomics_encoder")
    for name, parameter in built.named_parameters():
        if name.startswith(encoders):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


# --------------------------------------------------------------------------- #
# Shape validation
# --------------------------------------------------------------------------- #
def test_wrong_genomic_width_raises_rather_than_being_reshaped(
    model: MultimodalCancerModel,
) -> None:
    inputs = _inputs()
    inputs["genomics"] = torch.randn(BATCH_SIZE, PAM50_GENE_COUNT - 1)

    with pytest.raises(ValueError, match="genomic features"):
        model(**inputs)


def test_unbatched_genomic_input_raises(model: MultimodalCancerModel) -> None:
    inputs = _inputs()
    inputs["genomics"] = torch.randn(PAM50_GENE_COUNT)

    with pytest.raises(ValueError, match="batch_size"):
        model(**inputs)


def test_wrong_clinical_width_raises(model: MultimodalCancerModel) -> None:
    """The legacy 22-dimensional vector must not silently work."""
    inputs = _inputs()
    inputs["clinical"] = torch.randn(BATCH_SIZE, 22)

    with pytest.raises(RuntimeError):
        model(**inputs)


def test_forward_without_any_modality_raises(model: MultimodalCancerModel) -> None:
    with pytest.raises(ValueError, match="At least one modality"):
        model()


def test_supplying_a_disabled_modality_raises(
    small_model_config: dict[str, Any],
) -> None:
    built = build_model({**small_model_config, "enable_genomics": False})

    with pytest.raises(ValueError, match="genomic encoder is disabled"):
        built(**_inputs())


def test_disabling_survival_removes_the_risk_score(
    small_model_config: dict[str, Any],
) -> None:
    built = build_model({**small_model_config, "enable_survival": False})

    output = built(**_inputs())

    assert built.survival_head is None
    assert RISK_SCORE_KEY not in output


# --------------------------------------------------------------------------- #
# Integration with the dataset
# --------------------------------------------------------------------------- #
def test_a_real_dataset_batch_flows_through_the_model(
    synthetic_dataset: MultimodalDataset,
    model: MultimodalCancerModel,
) -> None:
    """dataset -> collate -> build_model_inputs -> forward, unchanged shapes."""
    batch = collate_multimodal([synthetic_dataset[i] for i in range(BATCH_SIZE)])
    inputs = build_model_inputs(batch, torch.device("cpu"))

    output = model(**inputs)

    assert inputs["clinical"].shape == (BATCH_SIZE, CLINICAL_FEATURE_DIM)
    assert inputs["genomics"].shape == (BATCH_SIZE, PAM50_GENE_COUNT)
    assert output["subtype_logits"].shape == (BATCH_SIZE, len(PAM50_SUBTYPES))
    assert output[RISK_SCORE_KEY].shape == (BATCH_SIZE, 1)


def test_genomic_transformer_rejects_an_indivisible_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GenomicTransformer(input_dim=PAM50_GENE_COUNT, d_model=16, nhead=3)


def test_genomic_transformer_rejects_a_non_positive_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        GenomicTransformer(input_dim=0)
