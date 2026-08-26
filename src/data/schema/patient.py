"""Top-level patient record schema for the active TCGA-BRCA pipeline.

Defines :class:`Patient`, the two-modality record composing clinical and
genomics data for a single patient, and :class:`BrcaTargets`, the per-task
target container that carries the per-task masking information.

Targets live on the patient record rather than in a parallel array so that
labels are aligned to a patient by identity, never by row position
(CLAUDE.md section 10).
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.pam50 import PAM50_SUBTYPES
from src.data.schema.clinical import ClinicalData
from src.data.schema.genomics import GenomicsData


class CancerType(str, Enum):
    """Cancer cohorts targeted by the active implementation."""

    BREAST = "breast"


class BrcaTargets(BaseModel):
    """Per-task targets for one patient, with explicit missingness.

    Every target is optional. ``None`` means "no usable label for this task",
    which the dataset converts into a task mask of ``0``. A missing label is
    never substituted with a default class (CLAUDE.md section 13).
    """

    model_config = ConfigDict(extra="forbid")

    subtype_index: Optional[int] = Field(default=None, ge=0, lt=len(PAM50_SUBTYPES))
    er_positive: Optional[bool] = None
    pr_positive: Optional[bool] = None
    her2_positive: Optional[bool] = None

    os_months: Optional[float] = Field(default=None, ge=0.0)
    os_event: Optional[bool] = None

    #: Set when a patient has survival values on file that are nonetheless
    #: unusable for the Cox objective (zero censored follow-up, unresolved
    #: cross-source conflict). Such patients remain fully available to the
    #: classification heads.
    survival_excluded: bool = False
    survival_exclusion_reason: Optional[str] = None

    @model_validator(mode="after")
    def check_exclusion_is_documented(self) -> "BrcaTargets":
        """Require a reason whenever survival is excluded, and vice versa."""
        if self.survival_excluded and not self.survival_exclusion_reason:
            raise ValueError(
                "survival_excluded=True requires survival_exclusion_reason."
            )
        if self.survival_exclusion_reason and not self.survival_excluded:
            raise ValueError(
                "survival_exclusion_reason set without survival_excluded=True."
            )
        return self

    @property
    def has_survival(self) -> bool:
        """Whether this patient contributes to the Cox partial likelihood.

        Requires a mapped event indicator and a strictly positive follow-up
        time, and that the record has not been explicitly excluded.
        """
        if self.survival_excluded:
            return False
        if self.os_months is None or self.os_event is None:
            return False
        return self.os_months > 0.0


class Patient(BaseModel):
    """Two-modality record for a single TCGA-BRCA patient.

    Composes clinical data with genomics records. Either modality may be
    absent; the dataset layer validates that a patient carries what the
    active model requires before yielding it.
    """

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    cancer_type: CancerType = CancerType.BREAST
    date_of_diagnosis: Optional[date] = None

    clinical: Optional[ClinicalData] = None
    genomics: list[GenomicsData] = Field(default_factory=list)
    targets: BrcaTargets = Field(default_factory=BrcaTargets)

    def available_modalities(self) -> list[str]:
        """Return the names of modalities populated on this patient.

        Returns:
            A list containing any of "clinical", "genomics", depending on
            which are present/non-empty.
        """
        mod: list[str] = []

        if self.clinical:
            mod.append("clinical")
        if self.genomics:
            mod.append("genomics")

        return mod
