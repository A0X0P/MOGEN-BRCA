"""Split integrity and determinism tests.

Two properties are mandatory (CLAUDE.md section 12): splits are at the
*patient* level with no overlap between partitions, and they are deterministic
given the same seed and the same cohort — independently of the order the cohort
was loaded in.
"""

from __future__ import annotations

from typing import Callable

import pytest

from src.data.schema.patient import Patient
from src.data.splits import (
    DataSplit,
    split_patients,
    stratum_distribution,
    subtype_event_key,
    verify_split_integrity,
)
from src.data.pam50 import PAM50_SUBTYPES
from tests.conftest import build_cohort

SEED = 42


@pytest.fixture
def split(synthetic_cohort: list[Patient]) -> DataSplit:
    """A stratified split of the synthetic cohort at the default seed."""
    return split_patients(synthetic_cohort, seed=SEED)


def _ids(split: DataSplit) -> dict[str, list[str]]:
    return split.patient_ids()


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #
def test_partitions_are_disjoint_and_cover_the_cohort(
    split: DataSplit,
    synthetic_cohort: list[Patient],
) -> None:
    ids = _ids(split)
    train, val, test = set(ids["train"]), set(ids["val"]), set(ids["test"])

    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == {p.patient_id for p in synthetic_cohort}
    assert sum(split.sizes().values()) == len(synthetic_cohort)


def test_no_partition_is_empty(split: DataSplit) -> None:
    assert all(size > 0 for size in split.sizes().values())


def test_verify_split_integrity_detects_an_overlapping_patient(
    synthetic_cohort: list[Patient],
) -> None:
    leaked = synthetic_cohort[0]

    with pytest.raises(ValueError, match="appears in both"):
        DataSplit(
            train=[leaked, synthetic_cohort[1]],
            val=[leaked],
            test=[synthetic_cohort[2]],
        )


def test_verify_split_integrity_detects_a_within_partition_duplicate(
    synthetic_cohort: list[Patient],
) -> None:
    duplicated = synthetic_cohort[0]
    split = DataSplit(
        train=[synthetic_cohort[1]],
        val=[synthetic_cohort[2]],
        test=[synthetic_cohort[3]],
    )
    split.test = [duplicated, duplicated]

    with pytest.raises(ValueError, match="appears in both"):
        verify_split_integrity(split)


def test_duplicate_patient_ids_are_rejected(
    make_patient: Callable[..., Patient],
) -> None:
    cohort = [make_patient(i, patient_id="SYNTH-DUP") for i in range(4)]

    with pytest.raises(ValueError, match="duplicated patient id"):
        split_patients(cohort, seed=SEED)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_and_cohort_give_the_same_split(
    synthetic_cohort: list[Patient],
) -> None:
    first = split_patients(synthetic_cohort, seed=SEED)
    second = split_patients(build_cohort(), seed=SEED)

    assert _ids(first) == _ids(second)


def test_split_is_independent_of_cohort_load_order(
    synthetic_cohort: list[Patient],
) -> None:
    """The cohort is sorted by id before shuffling, so order cannot leak in."""
    shuffled = list(reversed(synthetic_cohort))

    baseline = split_patients(synthetic_cohort, seed=SEED)
    reordered = split_patients(shuffled, seed=SEED)

    assert {name: sorted(ids) for name, ids in _ids(reordered).items()} == {
        name: sorted(ids) for name, ids in _ids(baseline).items()
    }


def test_different_seeds_give_different_splits(
    synthetic_cohort: list[Patient],
) -> None:
    first = split_patients(synthetic_cohort, seed=SEED)
    second = split_patients(synthetic_cohort, seed=SEED + 1)

    assert set(_ids(first)["test"]) != set(_ids(second)["test"])


# --------------------------------------------------------------------------- #
# Fractions and validation
# --------------------------------------------------------------------------- #
def test_unstratified_split_respects_the_requested_fractions(
    synthetic_cohort: list[Patient],
) -> None:
    total = len(synthetic_cohort)

    result = split_patients(
        synthetic_cohort,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=SEED,
        stratify=False,
    )

    assert result.sizes()["val"] == round(total * 0.2)
    assert result.sizes()["test"] == round(total * 0.2)
    assert result.sizes()["train"] == total - 2 * round(total * 0.2)


@pytest.mark.parametrize(
    ("val_fraction", "test_fraction"),
    [(0.0, 0.15), (0.15, 0.0), (-0.1, 0.15), (1.0, 0.15), (0.5, 0.5), (0.6, 0.5)],
)
def test_invalid_fractions_are_rejected(
    val_fraction: float,
    test_fraction: float,
    synthetic_cohort: list[Patient],
) -> None:
    with pytest.raises(ValueError):
        split_patients(
            synthetic_cohort,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=SEED,
        )


def test_a_cohort_too_small_to_split_is_rejected(
    make_patient: Callable[..., Patient],
) -> None:
    with pytest.raises(ValueError, match="at least 3 patients"):
        split_patients([make_patient(0), make_patient(1)], seed=SEED)


# --------------------------------------------------------------------------- #
# Stratification
# --------------------------------------------------------------------------- #
def test_stratification_spreads_populated_strata_across_partitions(
    split: DataSplit,
    synthetic_cohort: list[Patient],
) -> None:
    """Strata with at least three members must reach val and test too."""
    cohort_strata = stratum_distribution(synthetic_cohort)
    populated = {
        stratum for stratum, count in cohort_strata.items() if count >= 3
    }

    for name in ("val", "test"):
        present = set(stratum_distribution(split.partitions[name]))
        assert populated <= present, f"{name} is missing {populated - present}"


def test_missing_label_strata_are_represented_rather_than_dropped(
    synthetic_cohort: list[Patient],
) -> None:
    distribution = stratum_distribution(synthetic_cohort)

    assert any(key.startswith("subtype:missing") for key in distribution)
    assert any(key.endswith("survival:missing") for key in distribution)


def test_stratum_distribution_accounts_for_every_patient(
    synthetic_cohort: list[Patient],
) -> None:
    assert sum(stratum_distribution(synthetic_cohort).values()) == len(
        synthetic_cohort
    )


def test_stratum_key_reports_missing_labels_instead_of_inventing_a_class(
    make_patient: Callable[..., Patient],
) -> None:
    masked = make_patient(subtype_index=None, os_months=None, os_event=None)

    assert subtype_event_key(masked) == "subtype:missing|survival:missing"
    assert not any(name in subtype_event_key(masked) for name in PAM50_SUBTYPES)


@pytest.mark.parametrize(
    ("subtype_index", "os_event", "expected"),
    [
        (0, True, "Luminal A|event"),
        (3, False, "Basal-like|censored"),
        (2, True, "HER2-enriched|event"),
    ],
)
def test_stratum_key_crosses_subtype_with_event_status(
    subtype_index: int,
    os_event: bool,
    expected: str,
    make_patient: Callable[..., Patient],
) -> None:
    patient = make_patient(
        subtype_index=subtype_index, os_months=18.0, os_event=os_event
    )

    assert subtype_event_key(patient) == expected


def test_survival_excluded_patient_joins_the_survival_missing_stratum(
    make_patient: Callable[..., Patient],
) -> None:
    """It keeps its subtype stratum; only its survival contribution is dropped."""
    from tests.conftest import CONFLICT_REASON

    patient = make_patient(
        subtype_index=1,
        os_months=0.85,
        os_event=False,
        survival_excluded=True,
        survival_exclusion_reason=CONFLICT_REASON,
    )

    assert subtype_event_key(patient) == "Luminal B|survival:missing"
    assert patient.targets.has_survival is False
