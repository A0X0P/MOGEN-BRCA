"""Real-data contract tests for the ratified TCGA-BRCA cohort.

Every other test module works on synthetic patients so the suite runs without
the raw archives. This module is the one that reads the *actual* files named in
``configs/breast/data.yaml`` and pins the ratified data contract:

* the 1082-patient three-way intersection,
* the usable-label counts per task (981 / 1031 / 1028 / 937),
* the HER2 score-driven rule's class balance (769 negative / 168 positive),
* the documented survival exclusions and the resulting eligible count,
* the 50-gene panel in canonical order,
* and the independence of the receptor labels from the PAM50 subtype label.

The whole module skips when the raw sources are absent, so a checkout without
the ~1.5 GB archive still runs green.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from scripts import run_train
from src.data.brca_loader import (
    SURVIVAL_CONFLICT_EXCLUSIONS,
    ZERO_FOLLOWUP_REASON,
    CohortReport,
)
from src.data.datasets.multimodal_dataset import MultimodalDataset
from src.data.pam50 import (
    PAM50_GENE_COUNT,
    PAM50_GENES,
    PAM50_SUBTYPE_TO_INDEX,
    PAM50_SUBTYPES,
)
from src.data.schema.clinical import CLINICAL_FEATURE_DIM
from src.data.schema.genomics import OmicsType
from src.data.schema.patient import Patient
from src.data.splits import split_patients, verify_split_integrity
from src.data.tasks import CLASSIFICATION_TASKS
from src.utils.io import load_yaml
from tests.conftest import REPO_ROOT

DATA_CONFIG_PATH = REPO_ROOT / "configs" / "breast" / "data.yaml"

#: Ratified cohort size — the retained three-way patient intersection.
N_COHORT = 1082

#: Ratified usable-label counts for the four classification tasks.
RATIFIED_LABEL_COUNTS = {"subtype": 981, "er": 1031, "pr": 1028, "her2": 937}

#: Ratified HER2 class balance under the score-driven rule.
HER2_NEGATIVE = 769
HER2_POSITIVE = 168

#: Ratified observed deaths among survival-eligible patients.
N_EVENTS = 151

#: Censored patients with ``OS_MONTHS == 0``, excluded from the Cox objective.
N_ZERO_FOLLOWUP_EXCLUSIONS = 13

#: Survival-eligible patients once *both* ratified exclusions are applied:
#: 1082 - 13 zero-follow-up censored - 1 documented cross-source conflict.
#: Root CLAUDE.md section 29 quotes 1069, which applies the zero-follow-up rule
#: only; see :func:`test_survival_eligibility_reconciles_with_the_exclusions`.
SURVIVAL_ELIGIBLE = 1068
DOCUMENTED_SURVIVAL_COUNT = 1069

#: Provenance: rows in each source before intersection.
N_PAN_CAN = 1084
N_LEGACY = 1101
N_EXPRESSION = 1082


def _data_config() -> dict[str, Any]:
    return dict(load_yaml(DATA_CONFIG_PATH))


def _missing_sources() -> list[str]:
    """Paths named by the data config that are not on disk."""
    if not DATA_CONFIG_PATH.exists():
        return [str(DATA_CONFIG_PATH)]

    sources = _data_config().get("sources") or {}
    return [
        str(REPO_ROOT / relative)
        for relative in sources.values()
        if not (REPO_ROOT / relative).exists()
    ]


_MISSING = _missing_sources()

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"raw TCGA-BRCA sources are not present: {_MISSING}",
)


# --------------------------------------------------------------------------- #
# Fixtures — the cohort is loaded once for the module
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def loaded() -> tuple[list[Patient], CohortReport]:
    """The real cohort, loaded through the same helper the trainer uses."""
    return run_train.load_cohort(_data_config())


@pytest.fixture(scope="module")
def patients(loaded: tuple[list[Patient], CohortReport]) -> list[Patient]:
    return loaded[0]


@pytest.fixture(scope="module")
def report(loaded: tuple[list[Patient], CohortReport]) -> CohortReport:
    return loaded[1]


def _survival_exclusions(patients: list[Patient], reason: str) -> list[Patient]:
    return [p for p in patients if p.targets.survival_exclusion_reason == reason]


# --------------------------------------------------------------------------- #
# Cohort identity
# --------------------------------------------------------------------------- #
def test_cohort_size_matches_the_ratified_intersection(
    patients: list[Patient],
    report: CohortReport,
) -> None:
    assert report.n_cohort == N_COHORT
    assert len(patients) == N_COHORT


def test_source_row_counts_are_the_expected_provenance(report: CohortReport) -> None:
    """The intersection is a real narrowing, not an accidental identity."""
    assert report.n_pan_can == N_PAN_CAN
    assert report.n_legacy == N_LEGACY
    assert report.n_expression == N_EXPRESSION
    assert report.n_cohort <= min(N_PAN_CAN, N_LEGACY, N_EXPRESSION)


def test_patient_ids_are_unique_sorted_tcga_barcodes(patients: list[Patient]) -> None:
    ids = [patient.patient_id for patient in patients]

    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)
    assert all(patient_id.startswith("TCGA-") for patient_id in ids)
    assert all(len(patient_id) == 12 for patient_id in ids)


def test_every_patient_carries_both_modalities(patients: list[Patient]) -> None:
    """Two modalities, both mandatory: the cohort is the intersection."""
    assert all(patient.clinical is not None for patient in patients)

    for patient in patients:
        expression = [
            record
            for record in patient.genomics
            if record.omics_type is OmicsType.RNA_SEQ
        ]
        assert len(expression) == 1, patient.patient_id
        assert not [
            record
            for record in patient.genomics
            if record.omics_type is not OmicsType.RNA_SEQ
        ]


# --------------------------------------------------------------------------- #
# Genomic panel
# --------------------------------------------------------------------------- #
def test_expression_records_use_the_canonical_fifty_gene_panel(
    patients: list[Patient],
    report: CohortReport,
) -> None:
    assert report.genes == PAM50_GENE_COUNT == 50

    for patient in patients:
        record = patient.genomics[0]
        assert tuple(record.feature_ids) == PAM50_GENES, patient.patient_id
        assert len(record.values) == PAM50_GENE_COUNT


def test_expression_values_are_finite_and_non_negative(
    patients: list[Patient],
) -> None:
    """RSEM values are non-negative; a NaN would silently poison the encoder."""
    for patient in patients:
        for symbol, value in zip(patient.genomics[0].feature_ids, patient.genomics[0].values):
            assert math.isfinite(value), f"{patient.patient_id}/{symbol}"
            assert value >= 0.0, f"{patient.patient_id}/{symbol}"


# --------------------------------------------------------------------------- #
# Label counts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", sorted(RATIFIED_LABEL_COUNTS))
def test_usable_label_counts_match_the_data_contract(
    task: str,
    report: CohortReport,
) -> None:
    assert report.label_counts[task] == RATIFIED_LABEL_COUNTS[task]


def test_no_task_requires_every_patient(report: CohortReport) -> None:
    """Per-task masking, not complete-case filtering."""
    assert all(
        report.label_counts[task] < N_COHORT for task in RATIFIED_LABEL_COUNTS
    )
    assert len({report.label_counts[task] for task in RATIFIED_LABEL_COUNTS}) > 1


def test_subtype_labels_stay_inside_the_pam50_vocabulary(
    patients: list[Patient],
) -> None:
    observed = {
        patient.targets.subtype_index
        for patient in patients
        if patient.targets.subtype_index is not None
    }

    assert observed <= set(range(len(PAM50_SUBTYPES)))
    assert len(observed) == len(PAM50_SUBTYPES), "every PAM50 class should occur"


def test_dataset_masks_reproduce_the_report_counts(patients: list[Patient]) -> None:
    """The mask the model sees must equal the count the loader reported."""
    dataset = MultimodalDataset(patients=patients)

    counts = dataset.mask_counts()

    assert {task: counts[task] for task in RATIFIED_LABEL_COUNTS} == (
        RATIFIED_LABEL_COUNTS
    )
    assert counts["survival"] == SURVIVAL_ELIGIBLE
    assert dataset[0]["clinical"]["features"].shape == (CLINICAL_FEATURE_DIM,)
    assert dataset[0]["genomics"]["features"].shape == (PAM50_GENE_COUNT,)


# --------------------------------------------------------------------------- #
# HER2 policy
# --------------------------------------------------------------------------- #
def test_her2_class_balance_follows_the_score_driven_rule(
    patients: list[Patient],
) -> None:
    labels = [
        patient.targets.her2_positive
        for patient in patients
        if patient.targets.her2_positive is not None
    ]

    assert labels.count(False) == HER2_NEGATIVE
    assert labels.count(True) == HER2_POSITIVE
    assert len(labels) == RATIFIED_LABEL_COUNTS["her2"]


def test_her2_evidence_paths_account_for_every_patient(
    report: CohortReport,
) -> None:
    evidence = report.her2_evidence

    assert sum(evidence.values()) == N_COHORT
    masked = sum(count for path, count in evidence.items() if path.startswith("masked"))
    assert masked == N_COHORT - RATIFIED_LABEL_COUNTS["her2"]
    # The ratified rule resolves equivocal IHC with FISH, and lets FISH win an
    # outright conflict; both paths must actually be exercised by the data.
    assert evidence.get("fish-only", 0) > 0
    assert evidence.get("conflict:fish-wins", 0) > 0
    assert evidence.get("ihc-definitive", 0) > 0


# --------------------------------------------------------------------------- #
# Survival policy
# --------------------------------------------------------------------------- #
def test_survival_eligibility_reconciles_with_the_exclusions(
    patients: list[Patient],
    report: CohortReport,
) -> None:
    """Both ratified exclusions are applied, and the arithmetic is explicit.

    Root CLAUDE.md section 29 quotes 1069 survival observations, which is the
    count after the zero-follow-up rule alone. Applying the second ratified
    exclusion (the documented TCGA-E9-A245 conflict) removes one further
    patient, giving 1068. Nothing is dropped silently: both exclusions are
    recorded with a reason on the patient and in the cohort report.
    """
    zero_follow_up = _survival_exclusions(patients, ZERO_FOLLOWUP_REASON)
    conflicts = [
        patient
        for patient in patients
        if patient.patient_id in SURVIVAL_CONFLICT_EXCLUSIONS
    ]

    assert len(zero_follow_up) == N_ZERO_FOLLOWUP_EXCLUSIONS
    assert len(conflicts) == len(SURVIVAL_CONFLICT_EXCLUSIONS) == 1
    assert report.label_counts["survival"] == SURVIVAL_ELIGIBLE
    assert (
        N_COHORT - len(zero_follow_up) - len(conflicts)
        == report.label_counts["survival"]
    )
    # The section-29 figure, reconstructed from the same numbers.
    assert report.label_counts["survival"] + len(conflicts) == DOCUMENTED_SURVIVAL_COUNT


def test_every_survival_exclusion_carries_a_documented_reason(
    patients: list[Patient],
    report: CohortReport,
) -> None:
    excluded = [patient for patient in patients if patient.targets.survival_excluded]

    assert len(excluded) == N_COHORT - SURVIVAL_ELIGIBLE
    assert all(patient.targets.survival_exclusion_reason for patient in excluded)
    assert {patient.patient_id for patient in excluded} == set(
        report.survival_exclusions
    )
    assert all(patient.targets.has_survival is False for patient in excluded)


def test_zero_follow_up_exclusions_are_censored_observations(
    patients: list[Patient],
) -> None:
    """Only *censored* zero-time patients are excluded; a death is informative."""
    for patient in _survival_exclusions(patients, ZERO_FOLLOWUP_REASON):
        assert patient.targets.os_months == 0.0
        assert patient.targets.os_event is False


def test_observed_survival_times_are_preserved_not_rewritten(
    patients: list[Patient],
) -> None:
    """No epsilon is added to convert a zero follow-up into a positive time."""
    by_id = {patient.patient_id: patient for patient in patients}

    for patient_id in SURVIVAL_CONFLICT_EXCLUSIONS:
        targets = by_id[patient_id].targets
        assert targets.survival_excluded is True
        assert targets.os_months is not None and targets.os_months > 0.0

    zeros = _survival_exclusions(patients, ZERO_FOLLOWUP_REASON)
    assert all(patient.targets.os_months == 0.0 for patient in zeros)


def test_survival_excluded_patients_keep_their_classification_labels(
    patients: list[Patient],
) -> None:
    """The exclusion applies to the Cox objective only."""
    excluded = [patient for patient in patients if patient.targets.survival_excluded]

    retained = [
        patient
        for patient in excluded
        if any(
            getattr(patient.targets, field) is not None
            for field in ("subtype_index", "er_positive", "pr_positive", "her2_positive")
        )
    ]

    assert retained, "no survival-excluded patient kept a classification label"
    dataset = MultimodalDataset(patients=excluded)
    assert any(
        bool(dataset[index]["mask"][task])
        for index in range(len(dataset))
        for task in CLASSIFICATION_TASKS
    )
    assert not any(
        bool(dataset[index]["mask"]["survival"]) for index in range(len(dataset))
    )


def test_observed_event_count_matches_the_contract(
    patients: list[Patient],
    report: CohortReport,
) -> None:
    events = sum(
        bool(patient.targets.os_event)
        for patient in patients
        if patient.targets.has_survival
    )

    assert report.n_events == N_EVENTS == events


# --------------------------------------------------------------------------- #
# Label independence — receptors must never be read off the subtype
# --------------------------------------------------------------------------- #
def test_receptor_labels_are_not_derived_from_the_subtype_label(
    patients: list[Patient],
) -> None:
    """HER2-enriched subtype and HER2 receptor status are different variables."""
    her2_enriched = PAM50_SUBTYPE_TO_INDEX["HER2-enriched"]
    subtype_her2 = [
        patient
        for patient in patients
        if patient.targets.subtype_index == her2_enriched
    ]
    statuses = {
        patient.targets.her2_positive
        for patient in subtype_her2
        if patient.targets.her2_positive is not None
    }

    assert subtype_her2
    # If the label had been inferred from the subtype, this set would be {True}.
    assert statuses == {True, False}
    assert len(subtype_her2) != RATIFIED_LABEL_COUNTS["her2"]


def test_receptor_and_subtype_masks_come_from_independent_sources(
    patients: list[Patient],
) -> None:
    """Each label is absent for a different set of patients."""
    with_subtype = {
        patient.patient_id
        for patient in patients
        if patient.targets.subtype_index is not None
    }
    with_er = {
        patient.patient_id
        for patient in patients
        if patient.targets.er_positive is not None
    }

    assert with_subtype - with_er, "no patient has a subtype but no ER call"
    assert with_er - with_subtype, "no patient has an ER call but no subtype"


def test_hormone_receptor_labels_are_not_collapsed_into_each_other(
    patients: list[Patient],
) -> None:
    """ER and PR are separate binary tasks, resolved from separate columns."""
    disagreeing = [
        patient
        for patient in patients
        if patient.targets.er_positive is not None
        and patient.targets.pr_positive is not None
        and patient.targets.er_positive != patient.targets.pr_positive
    ]

    assert disagreeing


# --------------------------------------------------------------------------- #
# Splitting the real cohort
# --------------------------------------------------------------------------- #
def test_the_real_cohort_splits_at_the_patient_level(patients: list[Patient]) -> None:
    config = _data_config().get("split") or {}
    val_fraction = float(config.get("val_fraction", 0.15))
    test_fraction = float(config.get("test_fraction", 0.15))

    split = split_patients(
        patients,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=42,
        stratify=bool(config.get("stratify", True)),
    )
    verify_split_integrity(split)

    sizes = split.sizes()
    assert sum(sizes.values()) == N_COHORT
    assert all(size > 0 for size in sizes.values())
    assert sizes["val"] == pytest.approx(N_COHORT * val_fraction, rel=0.15)
    assert sizes["test"] == pytest.approx(N_COHORT * test_fraction, rel=0.15)
