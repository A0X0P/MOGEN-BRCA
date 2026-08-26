"""Dataset contract tests for the active two-modality TCGA-BRCA pipeline.

Covers the three things the dataset layer is responsible for:

1.  The 12-dimensional clinical encoding and its fixed column order.
2.  The 50-gene genomic vector, its canonical order, and the train-fold
    standardisation.
3.  Per-task masking — an absent label becomes
    :data:`~src.data.tasks.IGNORE_INDEX` with a ``False`` mask, and is never
    replaced by a default class or an invented survival time.
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch

from src.data.datasets.multimodal_dataset import (
    MultimodalDataset,
    extract_gene_vector,
    fit_gene_standardization,
)
from src.data.datasets.tabular_dataset import TabularDataset, fit_normalization_stats
from src.data.pam50 import PAM50_GENE_COUNT, PAM50_GENES
from src.data.schema.clinical import (
    CLINICAL_FEATURE_DIM,
    CLINICAL_FEATURE_NAMES,
    ClinicalData,
    NodalStage,
    Sex,
    TumorStage,
)
from src.data.schema.genomics import GenomicsData, OmicsType
from src.data.schema.patient import Patient
from src.data.tasks import CLASSIFICATION_TASKS, IGNORE_INDEX
from src.training.trainer import collate_multimodal
from tests.conftest import CONSTANT_GENE_INDEX, CONSTANT_GENE_VALUE, CONFLICT_REASON


# --------------------------------------------------------------------------- #
# Clinical encoding: 12 dimensions, fixed column order
# --------------------------------------------------------------------------- #
def test_clinical_contract_is_twelve_dimensional() -> None:
    assert CLINICAL_FEATURE_DIM == 12
    assert len(CLINICAL_FEATURE_NAMES) == 12


def test_clinical_column_order_is_the_ratified_layout() -> None:
    """1 age + 5 tumour stage + 5 nodal stage + 1 sex, in this exact order."""
    assert CLINICAL_FEATURE_NAMES == (
        "age",
        "stage_I",
        "stage_II",
        "stage_III",
        "stage_IV",
        "stage_Unknown",
        "nodal_N0",
        "nodal_N1",
        "nodal_N2",
        "nodal_N3",
        "nodal_Unknown",
        "sex_male",
    )


def test_clinical_encoding_produces_a_twelve_dim_float_tensor() -> None:
    record = ClinicalData(
        age=61,
        sex=Sex.FEMALE,
        tumor_stage=TumorStage.STAGE_II,
        nodal_stage=NodalStage.N1,
    )

    features = TabularDataset.encode(record)

    assert features.shape == (CLINICAL_FEATURE_DIM,)
    assert features.dtype is torch.float32


def test_clinical_encoding_one_hots_stage_nodal_and_sex() -> None:
    record = ClinicalData(
        age=70,
        sex=Sex.MALE,
        tumor_stage=TumorStage.STAGE_III,
        nodal_stage=NodalStage.N2,
    )

    features = TabularDataset.encode(record).tolist()
    columns = dict(zip(CLINICAL_FEATURE_NAMES, features))

    assert columns["age"] == 70.0
    assert columns["stage_III"] == 1.0
    assert columns["nodal_N2"] == 1.0
    assert columns["sex_male"] == 1.0
    # Exactly one stage, one nodal, and the sex indicator are set.
    assert sum(features[1:6]) == 1.0
    assert sum(features[6:11]) == 1.0


def test_unknown_stage_occupies_its_own_column_rather_than_being_imputed() -> None:
    record = ClinicalData(
        age=50,
        sex=Sex.FEMALE,
        tumor_stage=TumorStage.UNKNOWN,
        nodal_stage=NodalStage.UNKNOWN,
    )

    columns = dict(zip(CLINICAL_FEATURE_NAMES, TabularDataset.encode(record).tolist()))

    assert columns["stage_Unknown"] == 1.0
    assert columns["nodal_Unknown"] == 1.0
    assert columns["sex_male"] == 0.0


def test_age_is_z_scored_with_the_supplied_statistics() -> None:
    record = ClinicalData(
        age=60,
        sex=Sex.FEMALE,
        tumor_stage=TumorStage.STAGE_I,
        nodal_stage=NodalStage.N0,
    )
    stats = {"age": {"mean": 50.0, "std": 10.0}}

    features = TabularDataset.encode(record, stats)

    assert features[0].item() == pytest.approx(1.0)


def test_age_passes_through_without_statistics() -> None:
    record = ClinicalData(
        age=60,
        sex=Sex.FEMALE,
        tumor_stage=TumorStage.STAGE_I,
        nodal_stage=NodalStage.N0,
    )

    assert TabularDataset.encode(record)[0].item() == pytest.approx(60.0)


def test_zero_variance_age_statistics_do_not_divide_by_zero() -> None:
    record = ClinicalData(
        age=60,
        sex=Sex.FEMALE,
        tumor_stage=TumorStage.STAGE_I,
        nodal_stage=NodalStage.N0,
    )
    stats = {"age": {"mean": 60.0, "std": 0.0}}

    assert TabularDataset.encode(record, stats)[0].item() == pytest.approx(60.0)


def test_fit_normalization_stats_uses_population_std(
    make_patient: Callable[..., Patient],
) -> None:
    samples = [make_patient(i, age=age).clinical for i, age in enumerate((40, 60))]

    stats = fit_normalization_stats([s for s in samples if s is not None])

    assert stats["age"]["mean"] == pytest.approx(50.0)
    assert stats["age"]["std"] == pytest.approx(10.0)  # population, not sample


def test_fit_normalization_stats_rejects_an_empty_fold() -> None:
    with pytest.raises(ValueError):
        fit_normalization_stats([])


# --------------------------------------------------------------------------- #
# Genomic encoding: 50 genes, canonical order
# --------------------------------------------------------------------------- #
def test_gene_vector_has_the_panel_width_and_canonical_order(
    make_patient: Callable[..., Patient],
) -> None:
    """Values are read by symbol, so a reordered source row still aligns."""
    reversed_order = list(reversed(PAM50_GENES))
    values = [float(i) for i in range(PAM50_GENE_COUNT)]
    patient = make_patient(gene_order=reversed_order, values=values)

    vector = extract_gene_vector(patient)

    assert vector.shape == (PAM50_GENE_COUNT,)
    assert vector.tolist() == list(reversed(values))


def test_missing_gene_raises_rather_than_being_zero_filled(
    make_patient: Callable[..., Patient],
) -> None:
    """Zero is a legitimate expression value, so it cannot mean "absent"."""
    partial = PAM50_GENES[:-1]
    patient = make_patient(
        gene_order=partial,
        values=[1.0] * len(partial),
    )

    with pytest.raises(ValueError, match="missing"):
        extract_gene_vector(patient)


def test_patient_without_rna_seq_raises(make_patient: Callable[..., Patient]) -> None:
    patient = make_patient(with_genomics=False)

    with pytest.raises(ValueError, match="rna_seq"):
        extract_gene_vector(patient)


def test_non_rna_seq_omics_is_not_treated_as_expression(
    make_patient: Callable[..., Patient],
) -> None:
    patient = make_patient(with_genomics=False)
    patient.genomics.append(
        GenomicsData(
            omics_type=OmicsType.CNV,
            feature_ids=list(PAM50_GENES),
            values=[0.0] * PAM50_GENE_COUNT,
        )
    )

    with pytest.raises(ValueError, match="rna_seq"):
        extract_gene_vector(patient)


def test_gene_standardization_is_fitted_per_gene(
    synthetic_cohort: list[Patient],
) -> None:
    stats = fit_gene_standardization(synthetic_cohort)

    assert len(stats["mean"]) == PAM50_GENE_COUNT
    assert len(stats["std"]) == PAM50_GENE_COUNT
    # The constant gene has zero variance by construction.
    assert stats["std"][CONSTANT_GENE_INDEX] == pytest.approx(0.0)
    assert stats["mean"][CONSTANT_GENE_INDEX] == pytest.approx(CONSTANT_GENE_VALUE)


def test_fit_gene_standardization_rejects_an_empty_fold() -> None:
    with pytest.raises(ValueError):
        fit_gene_standardization([])


def test_standardized_genes_are_centred_and_scaled(
    synthetic_dataset: MultimodalDataset,
    synthetic_cohort: list[Patient],
    train_statistics: dict[str, object],
) -> None:
    stats = train_statistics["gene_standardization"]
    assert isinstance(stats, dict)

    raw = extract_gene_vector(synthetic_cohort[3])
    encoded = synthetic_dataset[3]["genomics"]["features"]

    gene = 7  # any gene with non-zero variance
    expected = (raw[gene].item() - stats["mean"][gene]) / stats["std"][gene]
    assert encoded[gene].item() == pytest.approx(expected, rel=1e-5)


def test_zero_variance_gene_is_passed_through_unscaled(
    synthetic_dataset: MultimodalDataset,
) -> None:
    encoded = synthetic_dataset[0]["genomics"]["features"]

    assert encoded[CONSTANT_GENE_INDEX].item() == pytest.approx(CONSTANT_GENE_VALUE)


def test_gene_statistics_of_the_wrong_width_are_rejected(
    synthetic_cohort: list[Patient],
) -> None:
    with pytest.raises(ValueError, match="entries"):
        MultimodalDataset(
            patients=synthetic_cohort,
            gene_standardization={"mean": [0.0], "std": [1.0]},
        )


def test_incomplete_gene_statistics_are_rejected(
    synthetic_cohort: list[Patient],
) -> None:
    with pytest.raises(ValueError, match="missing 'std'"):
        MultimodalDataset(
            patients=synthetic_cohort,
            gene_standardization={"mean": [0.0] * PAM50_GENE_COUNT},
        )


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def test_dataset_requires_at_least_one_patient() -> None:
    with pytest.raises(ValueError):
        MultimodalDataset(patients=[])


def test_dataset_rejects_a_patient_without_clinical_data(
    make_patient: Callable[..., Patient],
) -> None:
    with pytest.raises(ValueError, match="clinical"):
        MultimodalDataset(patients=[make_patient(with_clinical=False)])


def test_dataset_rejects_a_patient_without_expression_data(
    make_patient: Callable[..., Patient],
) -> None:
    with pytest.raises(ValueError, match="RNA-seq"):
        MultimodalDataset(patients=[make_patient(with_genomics=False)])


def test_dataset_sample_carries_both_modalities_and_the_mask_block(
    synthetic_dataset: MultimodalDataset,
) -> None:
    sample = synthetic_dataset[0]

    assert set(sample) == {
        "patient_id",
        "clinical",
        "genomics",
        "label",
        "mask",
        "survival",
    }
    assert sample["clinical"]["features"].shape == (CLINICAL_FEATURE_DIM,)
    assert sample["genomics"]["features"].shape == (PAM50_GENE_COUNT,)
    assert set(sample["mask"]) == {*CLASSIFICATION_TASKS, "survival"}


def test_dataset_length_and_ids_track_the_cohort(
    synthetic_dataset: MultimodalDataset,
    synthetic_cohort: list[Patient],
) -> None:
    assert len(synthetic_dataset) == len(synthetic_cohort)
    assert synthetic_dataset.patient_ids == [p.patient_id for p in synthetic_cohort]
    assert synthetic_dataset.gene_order == PAM50_GENES


# --------------------------------------------------------------------------- #
# Per-task masking
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", CLASSIFICATION_TASKS)
def test_missing_label_is_ignore_index_with_a_false_mask(
    task: str,
    make_patient: Callable[..., Patient],
) -> None:
    field = {
        "subtype": "subtype_index",
        "er": "er_positive",
        "pr": "pr_positive",
        "her2": "her2_positive",
    }[task]
    patient = make_patient(**{field: None})

    sample = MultimodalDataset(patients=[patient])[0]

    assert sample["label"][task].item() == IGNORE_INDEX
    assert bool(sample["mask"][task]) is False


@pytest.mark.parametrize(
    ("status", "expected_class"),
    [(False, 0), (True, 1)],
)
def test_receptor_status_maps_to_its_class_index(
    status: bool,
    expected_class: int,
    make_patient: Callable[..., Patient],
) -> None:
    sample = MultimodalDataset(patients=[make_patient(er_positive=status)])[0]

    assert sample["label"]["er"].item() == expected_class
    assert bool(sample["mask"]["er"]) is True


@pytest.mark.parametrize("subtype_index", [0, 1, 2, 3, 4])
def test_subtype_label_is_carried_through_unchanged(
    subtype_index: int,
    make_patient: Callable[..., Patient],
) -> None:
    sample = MultimodalDataset(patients=[make_patient(subtype_index=subtype_index)])[0]

    assert sample["label"]["subtype"].item() == subtype_index
    assert bool(sample["mask"]["subtype"]) is True


def test_mask_counts_match_the_labels_present_on_the_patients(
    synthetic_dataset: MultimodalDataset,
    synthetic_cohort: list[Patient],
) -> None:
    """Counted from the patient records, not from a hardcoded expectation."""
    expected = {
        "subtype": sum(p.targets.subtype_index is not None for p in synthetic_cohort),
        "er": sum(p.targets.er_positive is not None for p in synthetic_cohort),
        "pr": sum(p.targets.pr_positive is not None for p in synthetic_cohort),
        "her2": sum(p.targets.her2_positive is not None for p in synthetic_cohort),
        "survival": sum(p.targets.has_survival for p in synthetic_cohort),
    }

    assert synthetic_dataset.mask_counts() == expected


def test_tasks_do_not_share_a_mask(synthetic_dataset: MultimodalDataset) -> None:
    """Per-task masking, not complete-case filtering: the counts differ."""
    counts = synthetic_dataset.mask_counts()

    assert len(set(counts.values())) > 1
    assert all(count < len(synthetic_dataset) for count in counts.values())


# --------------------------------------------------------------------------- #
# Survival masking
# --------------------------------------------------------------------------- #
def test_usable_survival_is_passed_through(
    make_patient: Callable[..., Patient],
) -> None:
    patient = make_patient(os_months=31.5, os_event=True)

    sample = MultimodalDataset(patients=[patient])[0]

    assert bool(sample["mask"]["survival"]) is True
    assert sample["survival"]["duration"].item() == pytest.approx(31.5)
    assert sample["survival"]["event"].item() == pytest.approx(1.0)


def test_absent_survival_record_is_masked_with_placeholder_values(
    make_patient: Callable[..., Patient],
) -> None:
    """The placeholders exist only to keep the collated tensors rectangular."""
    patient = make_patient(os_months=None, os_event=None)

    sample = MultimodalDataset(patients=[patient])[0]

    assert bool(sample["mask"]["survival"]) is False
    assert sample["survival"]["duration"].item() == 0.0
    assert sample["survival"]["event"].item() == 0.0


def test_zero_follow_up_censored_patient_is_excluded_from_survival(
    make_patient: Callable[..., Patient],
) -> None:
    from src.data.brca_loader import ZERO_FOLLOWUP_REASON

    patient = make_patient(
        os_months=0.0,
        os_event=False,
        survival_excluded=True,
        survival_exclusion_reason=ZERO_FOLLOWUP_REASON,
    )

    sample = MultimodalDataset(patients=[patient])[0]

    assert bool(sample["mask"]["survival"]) is False


def test_documented_conflict_excludes_survival_but_keeps_classification(
    make_patient: Callable[..., Patient],
) -> None:
    """The exclusion applies to the Cox objective only."""
    patient = make_patient(
        subtype_index=3,
        er_positive=False,
        os_months=0.85,
        os_event=False,
        survival_excluded=True,
        survival_exclusion_reason=CONFLICT_REASON,
    )

    sample = MultimodalDataset(patients=[patient])[0]

    assert bool(sample["mask"]["survival"]) is False
    assert bool(sample["mask"]["subtype"]) is True
    assert sample["label"]["subtype"].item() == 3
    assert bool(sample["mask"]["er"]) is True
    assert sample["label"]["er"].item() == 0
    # The observed value is preserved on the record, not rewritten.
    assert patient.targets.os_months == pytest.approx(0.85)


def test_survival_exclusion_requires_a_documented_reason(
    make_patient: Callable[..., Patient],
) -> None:
    with pytest.raises(ValueError, match="survival_exclusion_reason"):
        make_patient(survival_excluded=True, survival_exclusion_reason=None)


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #
def test_collate_produces_rectangular_batches(
    synthetic_dataset: MultimodalDataset,
) -> None:
    batch = collate_multimodal([synthetic_dataset[i] for i in range(8)])

    assert batch["clinical"]["features"].shape == (8, CLINICAL_FEATURE_DIM)
    assert batch["genomics"]["features"].shape == (8, PAM50_GENE_COUNT)
    for task in CLASSIFICATION_TASKS:
        assert batch["label"][task].shape == (8,)
        assert batch["label"][task].dtype is torch.long
        assert batch["mask"][task].dtype is torch.bool
    assert batch["mask"]["survival"].shape == (8,)
    assert batch["survival"]["duration"].shape == (8,)
    assert batch["survival"]["event"].shape == (8,)


def test_collate_preserves_patient_identity_and_order(
    synthetic_dataset: MultimodalDataset,
) -> None:
    samples = [synthetic_dataset[i] for i in range(4)]

    batch = collate_multimodal(samples)

    assert batch["patient_id"] == [s["patient_id"] for s in samples]


def test_collate_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_multimodal([])


def test_collate_rejects_a_sample_missing_a_modality(
    synthetic_dataset: MultimodalDataset,
) -> None:
    sample = dict(synthetic_dataset[0])
    sample["genomics"] = None

    with pytest.raises(ValueError, match="genomics"):
        collate_multimodal([sample])
