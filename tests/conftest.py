"""Shared fixtures for the active two-modality TCGA-BRCA test suite.

The fixtures here build *synthetic* patients so that the unit tests do not
depend on the raw TCGA archives being present. They are deliberately built
through the real schema, preprocessing, and dataset code, so a contract change
— the 12-dimensional clinical vector, the 50-gene panel, per-task masking —
breaks these tests instead of silently passing.

Nothing here fabricates the ratified cohort statistics. The tests that assert
the real cohort numbers (N = 1082, 981/1031/1028/937 usable labels, 151 events)
read the actual files on disk and live in :mod:`tests.test_brca_cohort`.

Missingness is represented the same way the loader represents it: an absent
label is ``None`` on :class:`~src.data.schema.patient.BrcaTargets`, and a
survival record that exists but is unusable is flagged with
``survival_excluded`` plus a documented reason.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pytest

from src.data.brca_loader import ZERO_FOLLOWUP_REASON
from src.data.datasets.multimodal_dataset import (
    MultimodalDataset,
    fit_gene_standardization,
)
from src.data.datasets.tabular_dataset import fit_normalization_stats
from src.data.pam50 import PAM50_GENE_COUNT, PAM50_GENES
from src.data.schema.clinical import (
    CLINICAL_FEATURE_DIM,
    ClinicalData,
    NodalStage,
    Sex,
    TumorStage,
)
from src.data.schema.genomics import GenomicsData, OmicsType
from src.data.schema.patient import BrcaTargets, CancerType, Patient

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Reason recorded for the synthetic cohort's cross-source survival conflict.
#: The real one is :data:`~src.data.brca_loader.SURVIVAL_CONFLICT_EXCLUSIONS`.
CONFLICT_REASON = "synthetic unresolved overall-survival conflict"

#: Gene held constant across the synthetic cohort so tests can exercise the
#: zero-variance standardisation path without dividing by zero.
CONSTANT_GENE_INDEX = 0
CONSTANT_GENE_VALUE = 1.0

#: Size of the synthetic cohort. Large enough that the stratified split has
#: several populated strata, small enough for a fast training smoke test.
COHORT_SIZE = 120


def expression_values(index: int) -> list[float]:
    """Build a deterministic 50-gene expression vector for one patient.

    Values stand in for log1p-transformed RSEM expression, which is what the
    dataset layer expects to receive (the log1p step belongs to
    :class:`~src.data.preprocessing.genomics_preprocess.GenomicsPreprocessor`).

    Args:
        index: Patient index; seeds the generator so the vector is stable
            across runs and processes.

    Returns:
        A list of ``PAM50_GENE_COUNT`` non-negative floats.
    """
    rng = random.Random(1000 + index)
    values = [rng.uniform(0.0, 12.0) for _ in range(PAM50_GENE_COUNT)]
    values[CONSTANT_GENE_INDEX] = CONSTANT_GENE_VALUE
    return values


def build_patient(
    index: int = 0,
    *,
    patient_id: Optional[str] = None,
    age: int = 58,
    sex: Sex = Sex.FEMALE,
    tumor_stage: TumorStage = TumorStage.STAGE_II,
    nodal_stage: NodalStage = NodalStage.N0,
    subtype_index: Optional[int] = 0,
    er_positive: Optional[bool] = True,
    pr_positive: Optional[bool] = True,
    her2_positive: Optional[bool] = False,
    os_months: Optional[float] = 24.0,
    os_event: Optional[bool] = False,
    survival_excluded: bool = False,
    survival_exclusion_reason: Optional[str] = None,
    values: Optional[Sequence[float]] = None,
    gene_order: Sequence[str] = PAM50_GENES,
    with_clinical: bool = True,
    with_genomics: bool = True,
) -> Patient:
    """Build one synthetic :class:`~src.data.schema.patient.Patient`.

    Every argument has a valid default, so a test only states the field it is
    actually about. ``None`` for a target means "no usable label", exactly as
    the loader records it.

    Args:
        index: Patient index, used for the identifier and the expression seed.
        patient_id: Explicit identifier, overriding the index-derived one.
        age: Clinical age in years.
        sex: Clinical sex.
        tumor_stage: Collapsed pathological tumour stage.
        nodal_stage: Collapsed pathological nodal stage.
        subtype_index: PAM50 class index, or ``None`` when masked.
        er_positive: ER status, or ``None`` when masked.
        pr_positive: PR status, or ``None`` when masked.
        her2_positive: HER2 status, or ``None`` when masked.
        os_months: Overall-survival follow-up in months, or ``None``.
        os_event: Death indicator, or ``None``.
        survival_excluded: Whether the survival record is unusable for Cox.
        survival_exclusion_reason: Documented reason for the exclusion.
        values: Explicit expression values; defaults to
            :func:`expression_values`.
        gene_order: Gene symbols paired with ``values``.
        with_clinical: Whether to attach the clinical record.
        with_genomics: Whether to attach the RNA-seq record.

    Returns:
        A validated patient record.
    """
    clinical = (
        ClinicalData(
            age=age,
            sex=sex,
            tumor_stage=tumor_stage,
            nodal_stage=nodal_stage,
        )
        if with_clinical
        else None
    )

    genomics: list[GenomicsData] = []
    if with_genomics:
        genomics.append(
            GenomicsData(
                omics_type=OmicsType.RNA_SEQ,
                feature_ids=list(gene_order),
                values=list(values if values is not None else expression_values(index)),
            )
        )

    return Patient(
        patient_id=patient_id or f"SYNTH-{index:04d}",
        cancer_type=CancerType.BREAST,
        clinical=clinical,
        genomics=genomics,
        targets=BrcaTargets(
            subtype_index=subtype_index,
            er_positive=er_positive,
            pr_positive=pr_positive,
            her2_positive=her2_positive,
            os_months=os_months,
            os_event=os_event,
            survival_excluded=survival_excluded,
            survival_exclusion_reason=survival_exclusion_reason,
        ),
    )


#: Stage/nodal values cycled through the synthetic cohort, including the
#: explicit ``UNKNOWN`` categories that TCGA records as ``STAGE X`` / ``NX``.
TUMOR_STAGE_CYCLE: tuple[TumorStage, ...] = (
    TumorStage.STAGE_I,
    TumorStage.STAGE_II,
    TumorStage.STAGE_III,
    TumorStage.STAGE_IV,
    TumorStage.UNKNOWN,
)
NODAL_STAGE_CYCLE: tuple[NodalStage, ...] = (
    NodalStage.N0,
    NodalStage.N1,
    NodalStage.N2,
    NodalStage.N3,
    NodalStage.UNKNOWN,
)


def build_cohort(size: int = COHORT_SIZE) -> list[Patient]:
    """Build a synthetic cohort with realistic per-task missingness.

    Each task is missing for a different, coprime subset of the cohort, so no
    two tasks share a mask and complete-case filtering would visibly shrink the
    cohort. The survival records cover all four states the loader produces:
    usable, zero-follow-up censored, documented cross-source conflict, and no
    survival record at all.

    Args:
        size: Number of patients.

    Returns:
        A list of validated patients ordered by index.
    """
    patients: list[Patient] = []

    for index in range(size):
        os_months: Optional[float] = float(6 + index % 90)
        os_event: Optional[bool] = index % 4 == 0
        excluded = False
        reason: Optional[str] = None

        if index % 23 == 22:  # zero follow-up, censored: excluded from Cox only
            os_months, os_event, excluded, reason = (
                0.0,
                False,
                True,
                ZERO_FOLLOWUP_REASON,
            )
        elif index % 29 == 28:  # documented conflict: excluded from Cox only
            excluded, reason = True, CONFLICT_REASON
        elif index % 31 == 30:  # no survival record at all
            os_months, os_event = None, None

        patients.append(
            build_patient(
                index,
                age=30 + index % 50,
                sex=Sex.MALE if index % 60 == 59 else Sex.FEMALE,
                tumor_stage=TUMOR_STAGE_CYCLE[index % len(TUMOR_STAGE_CYCLE)],
                nodal_stage=NODAL_STAGE_CYCLE[index % len(NODAL_STAGE_CYCLE)],
                subtype_index=None if index % 10 == 9 else index % 5,
                er_positive=None if index % 11 == 10 else index % 2 == 0,
                pr_positive=None if index % 13 == 12 else index % 3 == 0,
                her2_positive=None if index % 7 == 6 else index % 5 == 0,
                os_months=os_months,
                os_event=os_event,
                survival_excluded=excluded,
                survival_exclusion_reason=reason,
            )
        )

    return patients


@pytest.fixture
def make_patient() -> Callable[..., Patient]:
    """Factory for one synthetic patient; see :func:`build_patient`."""
    return build_patient


@pytest.fixture
def synthetic_cohort() -> list[Patient]:
    """A synthetic cohort with per-task missingness; see :func:`build_cohort`."""
    return build_cohort()


@pytest.fixture
def train_statistics(synthetic_cohort: list[Patient]) -> dict[str, Any]:
    """Preprocessing statistics fitted on the synthetic cohort.

    In a real run these are fitted on the training fold only
    (:func:`scripts.run_train.fit_train_fold_statistics`). Here the whole
    synthetic cohort stands in for that fold; the leakage boundary itself is
    tested in :mod:`tests.test_training`.
    """
    return {
        "normalization_stats": fit_normalization_stats(
            [p.clinical for p in synthetic_cohort if p.clinical is not None]
        ),
        "gene_standardization": fit_gene_standardization(synthetic_cohort),
    }


@pytest.fixture
def synthetic_dataset(
    synthetic_cohort: list[Patient],
    train_statistics: dict[str, Any],
) -> MultimodalDataset:
    """The synthetic cohort as a fully configured dataset."""
    return MultimodalDataset(
        patients=synthetic_cohort,
        normalization_stats=train_statistics["normalization_stats"],
        gene_standardization=train_statistics["gene_standardization"],
    )


@pytest.fixture
def small_model_config() -> dict[str, Any]:
    """A narrow model config for fast forward/backward tests.

    Only the *widths* are reduced. The contract dimensions — the
    12-dimensional clinical input, the 50-gene genomic input, the 5-class
    subtype head, the binary receptor heads — are the ratified ones.
    """
    return {
        "embedding_dim": 32,
        "fused_dim": 32,
        "dropout": 0.0,
        "enable_clinical": True,
        "enable_genomics": True,
        "enable_survival": True,
        "clinical_input_dim": CLINICAL_FEATURE_DIM,
        "clinical_hidden_dim": 16,
        "clinical_num_blocks": 1,
        "genomics_input_dim": PAM50_GENE_COUNT,
        "genomics_d_model": 16,
        "genomics_nhead": 2,
        "genomics_num_layers": 1,
        "genomics_dim_feedforward": 32,
        "cross_attention_nhead": 2,
        "cross_attention_dropout": 0.0,
        "classification_hidden_dim": 16,
        "subtype_num_classes": 5,
        "receptor_num_classes": 2,
        "survival_hidden_dim": 16,
    }
