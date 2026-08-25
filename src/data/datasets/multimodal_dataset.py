"""PyTorch Dataset for the active two-modality TCGA-BRCA pipeline.

Combines the two active modalities — PAM50 gene expression and the
12-dimensional clinical vector — with the five per-task targets and their
masks.

Each sample is a dict:

    {
        "patient_id": str,
        "clinical":  {"features": FloatTensor(12,)},
        "genomics":  {"features": FloatTensor(50,)},
        "label":     {"subtype": long, "er": long, "pr": long, "her2": long},
        "mask":      {"subtype": bool, "er": bool, "pr": bool, "her2": bool,
                      "survival": bool},
        "survival":  {"duration": float, "event": float},
    }

Masking contract
----------------
A patient participates in a task only when that task's target is available.
Absent classification labels are written as :data:`~src.data.tasks.IGNORE_INDEX`
and their mask entry is ``False``; they are never replaced by a default class.
Absent survival values yield ``duration=0.0``, ``event=0.0`` with
``mask["survival"] = False`` — these placeholders exist only to keep the
collated tensors rectangular and are excluded from the Cox objective by the
mask.

Genomic preprocessing contract
------------------------------
Values arriving on :class:`~src.data.schema.genomics.GenomicsData` are assumed
already log1p-transformed by
:class:`~src.data.preprocessing.genomics_preprocess.GenomicsPreprocessor`.
This dataset applies only the gene-wise standardisation, whose statistics must
be fitted on the training partition via :func:`fit_gene_standardization`
(CLAUDE.md sections 4 and 12).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from src.data.pam50 import PAM50_GENES
from src.data.schema.genomics import GenomicsData, OmicsType
from src.data.schema.patient import Patient
from src.data.tasks import CLASSIFICATION_TASKS, IGNORE_INDEX
from src.data.datasets.tabular_dataset import TabularDataset

#: Omics modality consumed by the active pipeline.
_ACTIVE_OMICS_TYPE = OmicsType.RNA_SEQ


def extract_gene_vector(
    patient: Patient,
    gene_order: tuple[str, ...] = PAM50_GENES,
) -> torch.Tensor:
    """Extract one patient's expression vector in canonical gene order.

    Args:
        patient: Patient carrying at least one RNA-seq genomics record.
        gene_order: Canonical gene ordering. Defaults to
            :data:`~src.data.pam50.PAM50_GENES`.

    Returns:
        FloatTensor of shape ``(len(gene_order),)``.

    Raises:
        ValueError: If the patient has no RNA-seq record, or if any gene in
            ``gene_order`` is absent from it. Missing genes are never
            zero-filled, because a zero is a meaningful expression value.
    """
    record = _rna_seq_record(patient)
    if record is None:
        raise ValueError(
            f"Patient {patient.patient_id} has no {_ACTIVE_OMICS_TYPE.value} record."
        )

    by_symbol = dict(zip(record.feature_ids, record.values))
    missing = [gene for gene in gene_order if gene not in by_symbol]
    if missing:
        raise ValueError(
            f"Patient {patient.patient_id} is missing "
            f"{len(missing)} expression value(s): {missing[:5]}."
        )

    return torch.tensor(
        [float(by_symbol[gene]) for gene in gene_order], dtype=torch.float32
    )


def _rna_seq_record(patient: Patient) -> Optional[GenomicsData]:
    """Return the patient's RNA-seq record, or ``None`` when absent."""
    for record in patient.genomics:
        if record.omics_type is _ACTIVE_OMICS_TYPE:
            return record
    return None


def fit_gene_standardization(
    patients: list[Patient],
    gene_order: tuple[str, ...] = PAM50_GENES,
) -> dict[str, list[float]]:
    """Fit gene-wise mean/std on the TRAINING partition only.

    Args:
        patients: Training-partition patients.
        gene_order: Canonical gene ordering.

    Returns:
        Mapping with keys ``"mean"`` and ``"std"``, each a list of
        ``len(gene_order)`` floats aligned to ``gene_order``.

    Raises:
        ValueError: If ``patients`` is empty.
    """
    if not patients:
        raise ValueError("Cannot fit gene standardization on zero patients.")

    matrix = torch.stack(
        [extract_gene_vector(patient, gene_order) for patient in patients]
    )
    return {
        "mean": matrix.mean(dim=0).tolist(),
        # Population std: keeps a single-patient fold from producing NaN.
        "std": matrix.std(dim=0, unbiased=False).tolist(),
    }


class MultimodalDataset(Dataset):
    """Two-modality BRCA dataset yielding features, targets, and task masks.

    Every patient must carry both active modalities: the model has no
    modality-dropout path, so a patient missing clinical or genomic data is
    rejected at construction rather than silently zero-filled.
    """

    def __init__(
        self,
        patients: list[Patient],
        *,
        normalization_stats: Optional[dict[str, dict[str, float]]] = None,
        gene_standardization: Optional[dict[str, list[float]]] = None,
        gene_order: tuple[str, ...] = PAM50_GENES,
    ) -> None:
        """Initialise the dataset.

        Args:
            patients: Validated patients, each with clinical and genomics data.
            normalization_stats: Clinical z-score statistics from
                :func:`~src.data.datasets.tabular_dataset.fit_normalization_stats`,
                fitted on the training partition.
            gene_standardization: Gene-wise statistics from
                :func:`fit_gene_standardization`, fitted on the training
                partition.
            gene_order: Canonical gene ordering.

        Raises:
            ValueError: If ``patients`` is empty, if any patient lacks a
                required modality, or if the supplied gene statistics do not
                match ``gene_order``.
        """
        if not patients:
            raise ValueError("MultimodalDataset requires at least one patient.")

        self._patients = patients
        self._gene_order = gene_order
        self._normalization_stats = normalization_stats
        self._gene_standardization = gene_standardization

        self._validate_modalities()
        self._mean, self._std = self._resolve_gene_stats()

    def _validate_modalities(self) -> None:
        """Reject patients missing either active modality."""
        missing_clinical = [p.patient_id for p in self._patients if p.clinical is None]
        if missing_clinical:
            raise ValueError(
                f"{len(missing_clinical)} patient(s) lack clinical data, "
                f"e.g. {missing_clinical[:5]}."
            )

        missing_genomics = [
            p.patient_id for p in self._patients if _rna_seq_record(p) is None
        ]
        if missing_genomics:
            raise ValueError(
                f"{len(missing_genomics)} patient(s) lack RNA-seq data, "
                f"e.g. {missing_genomics[:5]}."
            )

    def _resolve_gene_stats(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Validate and materialise the gene standardisation tensors."""
        if self._gene_standardization is None:
            return None, None

        n_genes = len(self._gene_order)
        for key in ("mean", "std"):
            if key not in self._gene_standardization:
                raise ValueError(f"gene_standardization is missing '{key}'.")
            length = len(self._gene_standardization[key])
            if length != n_genes:
                raise ValueError(
                    f"gene_standardization['{key}'] has {length} entries, "
                    f"expected {n_genes}."
                )

        mean = torch.tensor(self._gene_standardization["mean"], dtype=torch.float32)
        std = torch.tensor(self._gene_standardization["std"], dtype=torch.float32)
        return mean, std

    def __len__(self) -> int:
        return len(self._patients)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return features, targets, and masks for one patient."""
        patient = self._patients[index]

        labels, masks = self._build_targets(patient)
        duration, event, survival_mask = self._build_survival(patient)
        masks["survival"] = survival_mask

        return {
            "patient_id": patient.patient_id,
            "clinical": {
                "features": TabularDataset.encode(
                    patient.clinical, self._normalization_stats
                )
            },
            "genomics": {"features": self._encode_genomics(patient)},
            "label": labels,
            "mask": masks,
            "survival": {"duration": duration, "event": event},
        }

    def _encode_genomics(self, patient: Patient) -> torch.Tensor:
        """Return the standardised PAM50 expression vector for one patient."""
        values = extract_gene_vector(patient, self._gene_order)

        if self._mean is None or self._std is None:
            return values

        # Zero-variance genes are passed through rather than divided by zero.
        scale = torch.where(self._std > 0, self._std, torch.ones_like(self._std))
        centered = torch.where(self._std > 0, values - self._mean, values)
        return centered / scale

    @staticmethod
    def _build_targets(
        patient: Patient,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build classification targets and their masks for one patient."""
        targets = patient.targets
        raw: dict[str, Optional[int]] = {
            "subtype": targets.subtype_index,
            "er": _bool_to_class(targets.er_positive),
            "pr": _bool_to_class(targets.pr_positive),
            "her2": _bool_to_class(targets.her2_positive),
        }

        labels: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for task in CLASSIFICATION_TASKS:
            value = raw[task]
            present = value is not None
            labels[task] = torch.tensor(
                value if present else IGNORE_INDEX, dtype=torch.long
            )
            masks[task] = torch.tensor(present, dtype=torch.bool)

        return labels, masks

    @staticmethod
    def _build_survival(
        patient: Patient,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build survival duration/event and mask for one patient."""
        targets = patient.targets
        usable = targets.has_survival

        duration = float(targets.os_months) if usable else 0.0
        event = float(targets.os_event) if usable else 0.0

        return (
            torch.tensor(duration, dtype=torch.float32),
            torch.tensor(event, dtype=torch.float32),
            torch.tensor(usable, dtype=torch.bool),
        )

    @property
    def gene_order(self) -> tuple[str, ...]:
        """The canonical gene ordering used for the genomic tensor."""
        return self._gene_order

    @property
    def patient_ids(self) -> list[str]:
        """Patient identifiers in dataset order."""
        return [patient.patient_id for patient in self._patients]

    def mask_counts(self) -> dict[str, int]:
        """Count usable labels per task across the dataset.

        Returns:
            Mapping of task name to the number of patients carrying a usable
            target for that task.
        """
        counts: dict[str, int] = {}
        for index in range(len(self)):
            for task, flag in self[index]["mask"].items():
                counts[task] = counts.get(task, 0) + int(bool(flag))
        return counts


def _bool_to_class(value: Optional[bool]) -> Optional[int]:
    """Map an optional boolean receptor status to a class index."""
    if value is None:
        return None
    return int(value)
