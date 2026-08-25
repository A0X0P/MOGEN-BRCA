"""Raw TCGA-BRCA sources → validated :class:`Patient` records.

This module is the single bridge between the files on disk and the typed data
contract. It implements the ratified cohort definition (CLAUDE.md sections 4-8)
and nothing else: no imputation of targets, no derivation of receptor status
from the PAM50 subtype, no substitution of survival times.

Sources
-------
1.  ``data_clinical_patient.txt`` (PanCancer Atlas) — age, sex, pathological
    tumour and nodal stage, PAM50 ``SUBTYPE``, and overall survival.
2.  ``brca_tcga_clinical_data.tsv`` (legacy cBioPortal export) — the only
    source carrying per-patient receptor assays (ER/PR IHC, HER2 IHC score,
    HER2 FISH).
3.  ``data_mrna_seq_v2_rsem.txt`` (PanCancer Atlas) — RSEM gene expression.

Cohort
------
The cohort is the patient-identifier intersection of the three sources, aligned
by identifier and never by row position. Expression columns are sample-level
(``TCGA-XX-XXXX-01``); the leading 12 characters give the patient identifier.

Targets
-------
Every target is optional and independently masked. A patient stays in the
cohort when a target is missing; only that task skips them.

HER2 follows the ratified score-driven rule: IHC 3 positive, IHC 0/1 negative,
IHC 2 or equivocal/indeterminate resolved by FISH, FISH winning any definitive
conflict, and masked when no usable evidence exists.

Survival excludes, from the survival task only, censored patients with
``OS_MONTHS == 0`` and the patient whose cross-source survival conflict remains
unresolved. Those patients are retained for every classification task.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

from src.data.pam50 import PAM50_GENES, resolve_gene_symbol, tcga_token_to_index
from src.data.preprocessing.genomics_preprocess import GenomicsPreprocessor
from src.data.preprocessing.tabular_preprocess import ClinicalPreprocessor
from src.data.schema.genomics import OmicsType
from src.data.schema.patient import BrcaTargets, CancerType, Patient
from src.utils.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Source layout
# --------------------------------------------------------------------------- #

#: Column names read from the PanCancer clinical patient file.
PAN_CAN_COLUMNS = MappingProxyType(
    {
        "patient_id": "PATIENT_ID",
        "age": "AGE",
        "sex": "SEX",
        "tumor_stage": "AJCC_PATHOLOGIC_TUMOR_STAGE",
        "nodal_stage": "PATH_N_STAGE",
        "subtype": "SUBTYPE",
        "os_months": "OS_MONTHS",
        "os_status": "OS_STATUS",
    }
)

#: Receptor columns read from the legacy clinical export. The ``Nte *`` columns
#: describe new tumour events, not the primary tumour, and are never read.
LEGACY_COLUMNS = MappingProxyType(
    {
        "patient_id": "Patient ID",
        "sample_id": "Sample ID",
        "er_ihc": "ER Status By IHC",
        "pr_ihc": "PR status by ihc",
        "her2_ihc_score": "HER2 ihc score",
        "her2_fish": "HER2 fish status",
        "her2_ihc_summary": "IHC-HER2",
    }
)

#: Sample suffix of the primary solid tumour, the only sample type used.
PRIMARY_TUMOR_SUFFIX = "-01"

#: Length of a TCGA patient barcode.
_PATIENT_BARCODE_LENGTH = 12

#: Tokens that mean "no value recorded" across the TCGA exports.
_NULL_TOKENS = frozenset(
    {"", "NA", "N/A", "[NOT AVAILABLE]", "[NOT EVALUATED]", "[UNKNOWN]", "NAN"}
)

#: Patients excluded from the survival task for a documented, non-numeric
#: reason. They remain available to every classification task.
SURVIVAL_CONFLICT_EXCLUSIONS = MappingProxyType(
    {
        "TCGA-E9-A245": (
            "unresolved overall-survival conflict between the PanCancer and "
            "legacy clinical sources"
        )
    }
)

#: Reason recorded for the zero-follow-up censored exclusion.
ZERO_FOLLOWUP_REASON = "censored observation with OS_MONTHS == 0"

#: Summary IHC tokens meaning the immunohistochemistry read was inconclusive.
#: Such a read is deferred to FISH regardless of any numeric score.
_HER2_NON_DEFINITIVE_IHC = frozenset({"EQUIVOCAL", "INDETERMINATE"})


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class CohortReport:
    """Provenance and reconciliation counts for one cohort load.

    Attributes:
        n_pan_can: Patients in the PanCancer clinical file.
        n_legacy: Patients in the legacy clinical export.
        n_expression: Patients with an expression column.
        n_cohort: Patients in the three-way intersection.
        label_counts: Usable-label count per task.
        n_events: Observed deaths among survival-eligible patients.
        survival_exclusions: Patient id → reason for survival-task exclusion.
        her2_evidence: Count of HER2 labels by the evidence path that resolved
            them.
        genes: Number of genes in the expression tensor.
    """

    n_pan_can: int = 0
    n_legacy: int = 0
    n_expression: int = 0
    n_cohort: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    n_events: int = 0
    survival_exclusions: dict[str, str] = field(default_factory=dict)
    her2_evidence: dict[str, int] = field(default_factory=dict)
    genes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the report."""
        return {
            "n_pan_can": self.n_pan_can,
            "n_legacy": self.n_legacy,
            "n_expression": self.n_expression,
            "n_cohort": self.n_cohort,
            "genes": self.genes,
            "label_counts": self.label_counts,
            "n_events": self.n_events,
            "survival_exclusions": self.survival_exclusions,
            "her2_evidence": self.her2_evidence,
        }

    def log(self) -> None:
        """Log the reconciliation counts at INFO level."""
        logger.info(
            "BRCA cohort: pan_can=%d legacy=%d expression=%d intersection=%d "
            "genes=%d",
            self.n_pan_can,
            self.n_legacy,
            self.n_expression,
            self.n_cohort,
            self.genes,
        )
        logger.info(
            "Usable labels: %s | events=%d | survival exclusions=%d",
            self.label_counts,
            self.n_events,
            len(self.survival_exclusions),
        )
        logger.info("HER2 evidence paths: %s", self.her2_evidence)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_brca_cohort(
    pan_can_clinical: str | Path,
    legacy_clinical: str | Path,
    expression: str | Path,
    gene_order: tuple[str, ...] = PAM50_GENES,
    survival_conflict_exclusions: Mapping[str, str] = SURVIVAL_CONFLICT_EXCLUSIONS,
) -> tuple[list[Patient], CohortReport]:
    """Load the TCGA-BRCA cohort as validated :class:`Patient` records.

    Args:
        pan_can_clinical: Path to ``data_clinical_patient.txt``.
        legacy_clinical: Path to ``brca_tcga_clinical_data.tsv``.
        expression: Path to ``data_mrna_seq_v2_rsem.txt``.
        gene_order: Canonical gene ordering for the expression vector.
        survival_conflict_exclusions: Patient id → reason, for patients removed
            from the survival task on documented data-conflict grounds.

    Returns:
        A tuple of (patients sorted by identifier, :class:`CohortReport`).

    Raises:
        FileNotFoundError: If a source file is absent.
        ValueError: If the intersection is empty, a required clinical value is
            unusable, or an expression value is missing or non-finite.
    """
    pan_can = _read_pan_can_clinical(Path(pan_can_clinical))
    legacy = _read_legacy_clinical(Path(legacy_clinical))
    expression_by_patient = _read_expression(Path(expression), gene_order)

    cohort_ids = sorted(set(pan_can) & set(legacy) & set(expression_by_patient))
    if not cohort_ids:
        raise ValueError(
            "The three clinical/expression sources share no patient identifiers."
        )

    report = CohortReport(
        n_pan_can=len(pan_can),
        n_legacy=len(legacy),
        n_expression=len(expression_by_patient),
        n_cohort=len(cohort_ids),
        genes=len(gene_order),
    )

    clinical_preprocessor = ClinicalPreprocessor()
    genomics_preprocessor = GenomicsPreprocessor(OmicsType.RNA_SEQ, list(gene_order))

    patients = [
        _build_patient(
            patient_id,
            pan_can[patient_id],
            legacy[patient_id],
            expression_by_patient[patient_id],
            clinical_preprocessor,
            genomics_preprocessor,
            survival_conflict_exclusions,
            report,
        )
        for patient_id in cohort_ids
    ]

    _summarise_labels(patients, report)
    report.log()
    return patients, report


def _build_patient(
    patient_id: str,
    pan_can_row: Mapping[str, str],
    legacy_row: Mapping[str, str],
    expression: Mapping[str, float],
    clinical_preprocessor: ClinicalPreprocessor,
    genomics_preprocessor: GenomicsPreprocessor,
    survival_conflict_exclusions: Mapping[str, str],
    report: CohortReport,
) -> Patient:
    """Assemble one patient's clinical vector, expression record, and targets."""
    clinical = clinical_preprocessor.process(
        {
            "age": pan_can_row[PAN_CAN_COLUMNS["age"]],
            "sex": pan_can_row[PAN_CAN_COLUMNS["sex"]],
            "tumor_stage": pan_can_row[PAN_CAN_COLUMNS["tumor_stage"]],
            "nodal_stage": pan_can_row[PAN_CAN_COLUMNS["nodal_stage"]],
        }
    )

    her2_positive, evidence = _resolve_her2(legacy_row)
    report.her2_evidence[evidence] = report.her2_evidence.get(evidence, 0) + 1

    os_months, os_event, exclusion = _resolve_survival(
        patient_id, pan_can_row, survival_conflict_exclusions
    )
    if exclusion is not None:
        report.survival_exclusions[patient_id] = exclusion

    return Patient(
        patient_id=patient_id,
        cancer_type=CancerType.BREAST,
        clinical=clinical,
        genomics=[genomics_preprocessor.process(expression)],
        targets=BrcaTargets(
            subtype_index=_resolve_subtype(pan_can_row[PAN_CAN_COLUMNS["subtype"]]),
            er_positive=_resolve_receptor(legacy_row[LEGACY_COLUMNS["er_ihc"]]),
            pr_positive=_resolve_receptor(legacy_row[LEGACY_COLUMNS["pr_ihc"]]),
            her2_positive=her2_positive,
            os_months=os_months,
            os_event=os_event,
            survival_excluded=exclusion is not None,
            survival_exclusion_reason=exclusion,
        ),
    )


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #
def _resolve_subtype(token: str) -> Optional[int]:
    """Map a ``SUBTYPE`` token to a PAM50 class index, or ``None`` if absent."""
    if _is_null(token):
        return None
    return tcga_token_to_index(token.strip())


def _resolve_receptor(value: str) -> Optional[bool]:
    """Map an IHC receptor call to a boolean, or ``None`` when unusable.

    ``Indeterminate`` is not a negative result, so it is masked rather than
    collapsed into the negative class.
    """
    token = value.strip().upper()
    if token == "POSITIVE":
        return True
    if token == "NEGATIVE":
        return False
    return None


def _resolve_her2(legacy_row: Mapping[str, str]) -> tuple[Optional[bool], str]:
    """Apply the ratified HER2 score-driven rule.

    Args:
        legacy_row: The patient's legacy clinical row.

    Returns:
        A tuple of (HER2 status or ``None`` when masked, evidence-path label
        recorded in the cohort report).
    """
    ihc = _her2_ihc_call(legacy_row)
    fish = _resolve_receptor(legacy_row[LEGACY_COLUMNS["her2_fish"]])

    if ihc is None:
        if fish is None:
            return None, "masked:no-evidence"
        return fish, "fish-only"

    if fish is None:
        return ihc, "ihc-definitive"

    if fish != ihc:
        # A definitive IHC/FISH disagreement is resolved in favour of FISH.
        return fish, "conflict:fish-wins"

    return ihc, "ihc-and-fish-agree"


def _her2_ihc_call(legacy_row: Mapping[str, str]) -> Optional[bool]:
    """Return the definitive IHC call, or ``None`` when it needs FISH.

    Precedence follows the ratified rule (CLAUDE.md section 7):

    1.  An ``Equivocal``/``Indeterminate`` summary read means the
        immunohistochemistry was not conclusive, so it defers to FISH *even
        when a numeric score is also recorded* — a score cannot be treated as
        definitive while the reporting summary records the read as
        indeterminate.
    2.  Otherwise the numeric score decides: 3 positive, 0 and 1 negative, 2
        equivocal and deferred to FISH.
    3.  The summary column decides only when no numeric score exists.

    Step 1 is what makes this rule reproduce the ratified label counts; the one
    patient it affects is documented in the migration report.
    """
    summary = legacy_row[LEGACY_COLUMNS["her2_ihc_summary"]]

    if summary.strip().upper() in _HER2_NON_DEFINITIVE_IHC:
        return None

    score = legacy_row[LEGACY_COLUMNS["her2_ihc_score"]].strip()

    if score in {"0", "1"}:
        return False
    if score == "3":
        return True
    if score == "2":
        return None

    return _resolve_receptor(summary)


def _resolve_survival(
    patient_id: str,
    pan_can_row: Mapping[str, str],
    survival_conflict_exclusions: Mapping[str, str],
) -> tuple[Optional[float], Optional[bool], Optional[str]]:
    """Resolve overall survival and any survival-task exclusion.

    Args:
        patient_id: TCGA patient barcode.
        pan_can_row: The patient's PanCancer clinical row.
        survival_conflict_exclusions: Documented conflict exclusions.

    Returns:
        A tuple of (OS months or ``None``, event flag or ``None``, exclusion
        reason or ``None``). The observed values are preserved even when the
        patient is excluded, so nothing is silently rewritten.

    Raises:
        ValueError: If ``OS_MONTHS`` is present but not a finite number.
    """
    months_token = pan_can_row[PAN_CAN_COLUMNS["os_months"]].strip()
    status_token = pan_can_row[PAN_CAN_COLUMNS["os_status"]].strip()

    if _is_null(months_token) or _is_null(status_token):
        return None, None, None

    try:
        os_months = float(months_token)
    except ValueError as exc:
        raise ValueError(
            f"{patient_id}: OS_MONTHS={months_token!r} is not a number."
        ) from exc

    if not math.isfinite(os_months):
        raise ValueError(f"{patient_id}: OS_MONTHS={months_token!r} is not finite.")

    os_event = status_token.upper().startswith("1")

    if patient_id in survival_conflict_exclusions:
        return os_months, os_event, survival_conflict_exclusions[patient_id]

    # A censored observation at time zero contributes no risk-set information
    # and is excluded rather than shifted by an artificial epsilon.
    if os_months == 0.0 and not os_event:
        return os_months, os_event, ZERO_FOLLOWUP_REASON

    return os_months, os_event, None


def _summarise_labels(patients: Iterable[Patient], report: CohortReport) -> None:
    """Fill the report's usable-label counts and event total."""
    counts = {"subtype": 0, "er": 0, "pr": 0, "her2": 0, "survival": 0}
    events = 0

    for patient in patients:
        targets = patient.targets
        counts["subtype"] += targets.subtype_index is not None
        counts["er"] += targets.er_positive is not None
        counts["pr"] += targets.pr_positive is not None
        counts["her2"] += targets.her2_positive is not None
        if targets.has_survival:
            counts["survival"] += 1
            events += bool(targets.os_event)

    report.label_counts = counts
    report.n_events = events


# --------------------------------------------------------------------------- #
# Source readers
# --------------------------------------------------------------------------- #
def _read_pan_can_clinical(path: Path) -> dict[str, dict[str, str]]:
    """Read the PanCancer clinical patient file, keyed by patient id.

    The cBioPortal format prefixes five metadata lines with ``#``; they are
    skipped so the real header is parsed.
    """
    rows = _read_delimited(path, skip_comments=True)
    return _index_by(path, rows, PAN_CAN_COLUMNS["patient_id"])


def _read_legacy_clinical(path: Path) -> dict[str, dict[str, str]]:
    """Read the legacy clinical export, keyed by patient id.

    A few patients carry both a primary (``-01``) and a metastatic (``-06``)
    sample row. The primary row is used; when several rows exist for a patient
    their receptor columns must agree, otherwise the conflict is raised rather
    than resolved arbitrarily.
    """
    rows = _read_delimited(path, skip_comments=False)
    by_patient: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        patient_id = row[LEGACY_COLUMNS["patient_id"]].strip()
        if patient_id:
            by_patient.setdefault(patient_id, []).append(row)

    return {
        patient_id: _select_legacy_row(patient_id, candidates)
        for patient_id, candidates in by_patient.items()
    }


def _select_legacy_row(
    patient_id: str, candidates: list[dict[str, str]]
) -> dict[str, str]:
    """Pick the primary-tumour row, asserting the receptor columns agree.

    Raises:
        ValueError: If duplicate rows disagree on any receptor column.
    """
    if len(candidates) == 1:
        return candidates[0]

    receptor_keys = ("er_ihc", "pr_ihc", "her2_ihc_score", "her2_fish", "her2_ihc_summary")
    distinct = {
        tuple(row[LEGACY_COLUMNS[key]].strip() for key in receptor_keys)
        for row in candidates
    }

    if len(distinct) > 1:
        raise ValueError(
            f"{patient_id}: duplicate legacy clinical rows disagree on receptor "
            f"status: {sorted(distinct)}."
        )

    primary = [
        row
        for row in candidates
        if row[LEGACY_COLUMNS["sample_id"]].strip().endswith(PRIMARY_TUMOR_SUFFIX)
    ]
    return primary[0] if primary else candidates[0]


def _read_expression(
    path: Path, gene_order: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Read the RSEM matrix, keeping only ``gene_order`` rows.

    The file is streamed so the 150 MB matrix is never held in memory. Row
    symbols are resolved through
    :func:`~src.data.pam50.resolve_gene_symbol`, so a matrix still using the
    deprecated published symbols (``KNTC2``, ``CDCA1``) is consumed correctly.
    Values are returned untransformed; the ``log1p`` step belongs to
    :class:`~src.data.preprocessing.genomics_preprocess.GenomicsPreprocessor`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a gene is absent or duplicated, if a value is blank or
            non-finite, or if two samples map to the same patient.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expression matrix not found: {path}")

    wanted = set(gene_order)
    by_patient: dict[str, dict[str, float]] = {}
    seen_genes: set[str] = set()

    with path.open(encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        patient_ids = _expression_patient_ids(header[2:])
        by_patient = {patient_id: {} for patient_id in patient_ids}

        for line in handle:
            fields = line.rstrip("\n").split("\t")
            symbol = resolve_gene_symbol(fields[0])
            if symbol is None or symbol not in wanted:
                continue
            if symbol in seen_genes:
                raise ValueError(f"Gene {symbol} appears more than once in {path}.")
            seen_genes.add(symbol)

            _assign_gene_row(symbol, fields[2:], patient_ids, by_patient, path)

    missing = sorted(wanted - seen_genes)
    if missing:
        raise ValueError(
            f"{len(missing)} gene(s) from the canonical panel are absent from "
            f"{path}: {missing}."
        )

    return by_patient


def _assign_gene_row(
    symbol: str,
    values: list[str],
    patient_ids: list[str],
    by_patient: dict[str, dict[str, float]],
    path: Path,
) -> None:
    """Write one gene's values into each patient's expression map.

    Raises:
        ValueError: If the row is short, or a value is blank or non-finite.
            A blank RSEM value is never treated as zero, because zero is a
            meaningful expression level.
    """
    if len(values) != len(patient_ids):
        raise ValueError(
            f"{path}: gene {symbol} has {len(values)} values for "
            f"{len(patient_ids)} samples."
        )

    for patient_id, raw in zip(patient_ids, values):
        token = raw.strip()
        if _is_null(token):
            raise ValueError(f"{path}: {patient_id} has no value for gene {symbol}.")

        value = float(token)
        if not math.isfinite(value):
            raise ValueError(
                f"{path}: {patient_id} has non-finite value {token!r} for gene "
                f"{symbol}."
            )
        by_patient[patient_id][symbol] = value


def _expression_patient_ids(sample_columns: list[str]) -> list[str]:
    """Map expression sample columns to patient identifiers.

    Raises:
        ValueError: If a column is not a TCGA barcode, or two columns map to
            the same patient (which would make the alignment ambiguous).
    """
    patient_ids: list[str] = []
    seen: set[str] = set()

    for column in sample_columns:
        sample_id = column.strip()
        if len(sample_id) < _PATIENT_BARCODE_LENGTH:
            raise ValueError(f"Expression column {sample_id!r} is not a TCGA barcode.")

        patient_id = sample_id[:_PATIENT_BARCODE_LENGTH]
        if patient_id in seen:
            raise ValueError(
                f"Two expression columns map to patient {patient_id}; patient "
                "alignment would be ambiguous."
            )

        seen.add(patient_id)
        patient_ids.append(patient_id)

    return patient_ids


def _read_delimited(path: Path, skip_comments: bool) -> list[dict[str, str]]:
    """Read a tab-delimited table into a list of row dicts.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Clinical file not found: {path}")

    with path.open(encoding="utf-8", newline="") as handle:
        lines = [
            line
            for line in handle
            if not (skip_comments and line.startswith("#"))
        ]

    return list(csv.DictReader(lines, delimiter="\t"))


def _index_by(
    path: Path, rows: list[dict[str, str]], key: str
) -> dict[str, dict[str, str]]:
    """Index rows by a unique key column.

    Raises:
        ValueError: If the column is absent or a key repeats.
    """
    if rows and key not in rows[0]:
        raise ValueError(f"{path}: expected column {key!r}, found {sorted(rows[0])}.")

    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row[key].strip()
        if not value:
            continue
        if value in indexed:
            raise ValueError(f"{path}: duplicate {key}={value!r}.")
        indexed[value] = row

    return indexed


def _is_null(token: str) -> bool:
    """Whether a raw token means "no value recorded"."""
    return token.strip().upper() in _NULL_TOKENS
