"""Patient-level train/val/test splitting for the TCGA-BRCA cohort.

Splits are deterministic given the same seed and the same set of patients: the
input is sorted by patient identifier before shuffling, so the result does not
depend on the order in which the cohort was loaded.

Stratification uses the PAM50 subtype crossed with survival-event status, not
``cancer_type`` — the active cohort is a single cancer type, so a cancer-type
stratification would be degenerate. Patients whose subtype or survival target is
absent form their own strata, which keeps missingness evenly spread across the
partitions instead of concentrated in one of them.

Patient-level integrity is mandatory (CLAUDE.md section 12): a patient appears
in exactly one partition, and duplicate identifiers are rejected outright.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from src.data.pam50 import PAM50_SUBTYPES
from src.data.schema.patient import Patient
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Signature of a stratification-key function.
StratificationKey = Callable[[Patient], str]


@dataclass
class DataSplit:
    """Container for train/val/test Patient lists.

    Attributes:
        train: Training patients.
        val: Validation patients.
        test: Test patients.
    """

    train: list[Patient]
    val: list[Patient]
    test: list[Patient]

    def __post_init__(self) -> None:
        verify_split_integrity(self)
        logger.info(
            "Split: train=%d val=%d test=%d (total=%d).",
            len(self.train),
            len(self.val),
            len(self.test),
            len(self.train) + len(self.val) + len(self.test),
        )

    @property
    def partitions(self) -> dict[str, list[Patient]]:
        """The three partitions keyed by name."""
        return {"train": self.train, "val": self.val, "test": self.test}

    def patient_ids(self) -> dict[str, list[str]]:
        """Patient identifiers per partition, in partition order."""
        return {
            name: [patient.patient_id for patient in patients]
            for name, patients in self.partitions.items()
        }

    def sizes(self) -> dict[str, int]:
        """Number of patients per partition."""
        return {name: len(patients) for name, patients in self.partitions.items()}


def subtype_event_key(patient: Patient) -> str:
    """Stratification key: PAM50 subtype crossed with survival-event status.

    Args:
        patient: Patient carrying BRCA targets.

    Returns:
        A stratum label such as ``"Basal-like|event"`` or
        ``"subtype:missing|censored"``.
    """
    targets = patient.targets

    if targets.subtype_index is None:
        subtype = "subtype:missing"
    else:
        subtype = PAM50_SUBTYPES[targets.subtype_index]

    if not targets.has_survival:
        outcome = "survival:missing"
    elif targets.os_event:
        outcome = "event"
    else:
        outcome = "censored"

    return f"{subtype}|{outcome}"


def split_patients(
    patients: list[Patient],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
    key_fn: StratificationKey = subtype_event_key,
) -> DataSplit:
    """Split patients into train/val/test partitions at the patient level.

    Args:
        patients: Full cohort of validated patients.
        val_fraction: Fraction of the cohort held out for validation.
        test_fraction: Fraction of the cohort held out for test.
        seed: Random seed; the same seed and cohort always give the same split.
        stratify: Whether to preserve the ``key_fn`` distribution across
            partitions.
        key_fn: Stratification-key function. Defaults to
            :func:`subtype_event_key`.

    Returns:
        A :class:`DataSplit`.

    Raises:
        ValueError: If the fractions are invalid, the cohort is smaller than
            three patients, or patient identifiers are duplicated.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be in (0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction + test_fraction = {val_fraction + test_fraction:.2f} "
            "must be less than 1.0."
        )
    if len(patients) < 3:
        raise ValueError(f"Need at least 3 patients to split, got {len(patients)}.")

    _reject_duplicate_ids(patients)

    # Sorting first makes the split independent of cohort load order.
    ordered = sorted(patients, key=lambda patient: patient.patient_id)
    rng = random.Random(seed)

    if not stratify:
        train, val, test = _cut(ordered, val_fraction, test_fraction, rng)
        return DataSplit(train=train, val=val, test=test)

    return _stratified_split(ordered, val_fraction, test_fraction, rng, key_fn)


def _stratified_split(
    patients: list[Patient],
    val_fraction: float,
    test_fraction: float,
    rng: random.Random,
    key_fn: StratificationKey,
) -> DataSplit:
    """Split each stratum independently, then concatenate the partitions."""
    by_stratum: dict[str, list[Patient]] = defaultdict(list)
    for patient in patients:
        by_stratum[key_fn(patient)].append(patient)

    train: list[Patient] = []
    val: list[Patient] = []
    test: list[Patient] = []

    for stratum in sorted(by_stratum):
        group = by_stratum[stratum]

        if len(group) < 3:
            logger.warning(
                "Stratum '%s' has only %d patient(s) — assigned to train.",
                stratum,
                len(group),
            )
            train.extend(group)
            continue

        group_train, group_val, group_test = _cut(
            group, val_fraction, test_fraction, rng
        )
        train.extend(group_train)
        val.extend(group_val)
        test.extend(group_test)

    return DataSplit(train=train, val=val, test=test)


def _cut(
    patients: list[Patient],
    val_fraction: float,
    test_fraction: float,
    rng: random.Random,
) -> tuple[list[Patient], list[Patient], list[Patient]]:
    """Shuffle one group and cut it into train/val/test lists.

    Raises:
        ValueError: If the requested hold-outs would leave no training data.
    """
    shuffled = list(patients)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, round(n * test_fraction))
    n_val = max(1, round(n * val_fraction))

    if n_test + n_val >= n:
        raise ValueError(
            f"Cannot hold out {n_test} test + {n_val} val patients from a group "
            f"of {n}."
        )

    return (
        shuffled[n_test + n_val :],
        shuffled[n_test : n_test + n_val],
        shuffled[:n_test],
    )


def verify_split_integrity(split: DataSplit) -> None:
    """Assert that no patient appears in more than one partition.

    Args:
        split: The split to check.

    Raises:
        ValueError: If any patient identifier occurs in two partitions or twice
            within one partition.
    """
    seen: dict[str, str] = {}

    for name, patients in split.partitions.items():
        for patient in patients:
            previous = seen.get(patient.patient_id)
            if previous is not None:
                raise ValueError(
                    f"Patient {patient.patient_id} appears in both '{previous}' "
                    f"and '{name}'; splits must be patient-level and disjoint."
                )
            seen[patient.patient_id] = name


def stratum_distribution(
    patients: Iterable[Patient],
    key_fn: StratificationKey = subtype_event_key,
) -> dict[str, int]:
    """Count patients per stratum, for logging a split's composition."""
    return dict(sorted(Counter(key_fn(patient) for patient in patients).items()))


def _reject_duplicate_ids(patients: list[Patient]) -> None:
    """Raise when the cohort contains duplicate patient identifiers."""
    duplicates = [
        patient_id
        for patient_id, count in Counter(p.patient_id for p in patients).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(
            f"Cohort contains {len(duplicates)} duplicated patient id(s), "
            f"e.g. {sorted(duplicates)[:5]}."
        )
