"""Clinical/tabular data schema for a TCGA-BRCA patient record.

Defines the structured clinical fields of the active two-modality BRCA
pipeline. The clinical contract is exactly four variables — age, sex,
pathological tumour stage, pathological nodal stage — which the dataset layer
encodes into a 12-dimensional vector.

Validation here is limited to structural/type correctness. Mapping raw TCGA
tokens (``"STAGE IIA"``, ``"N1MI"``, ``"NX"``, ...) onto these enums is the
responsibility of :mod:`src.data.preprocessing.tabular_preprocess`, not this
schema.

Missingness is represented explicitly by the ``UNKNOWN`` members of
:class:`TumorStage` and :class:`NodalStage` rather than by ``None``: an absent
stage is a real, informative category in TCGA (``STAGE X`` / ``NX``) and is
carried into the model as its own one-hot column.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field


class Sex(str, Enum):
    """Biological sex as recorded in the source clinical data."""

    MALE = "M"
    FEMALE = "F"


class TumorStage(str, Enum):
    """Collapsed AJCC pathological tumour stage.

    TCGA substages (``IA``, ``IIB``, ``IIIC``, ...) are collapsed to their
    parent stage by the preprocessing layer. ``UNKNOWN`` covers ``STAGE X``
    and blank values.
    """

    STAGE_I = "I"
    STAGE_II = "II"
    STAGE_III = "III"
    STAGE_IV = "IV"
    UNKNOWN = "Unknown"


class NodalStage(str, Enum):
    """Collapsed AJCC pathological nodal (N) stage.

    TCGA subcategories are collapsed to their parent category by the
    preprocessing layer. ``UNKNOWN`` covers ``NX`` and blank values.
    """

    N0 = "N0"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    UNKNOWN = "Unknown"


#: Deterministic one-hot column order for tumour stage. Do not reorder.
TUMOR_STAGE_ORDER: Final[tuple[TumorStage, ...]] = (
    TumorStage.STAGE_I,
    TumorStage.STAGE_II,
    TumorStage.STAGE_III,
    TumorStage.STAGE_IV,
    TumorStage.UNKNOWN,
)

#: Deterministic one-hot column order for nodal stage. Do not reorder.
NODAL_STAGE_ORDER: Final[tuple[NodalStage, ...]] = (
    NodalStage.N0,
    NodalStage.N1,
    NodalStage.N2,
    NodalStage.N3,
    NodalStage.UNKNOWN,
)

#: Names of the 12 clinical feature columns, in the exact order produced by
#: :class:`~src.data.datasets.tabular_dataset.ClinicalTabularDataset`. This is
#: the single source of truth for the clinical feature layout.
CLINICAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "age",
    *(f"stage_{stage.value}" for stage in TUMOR_STAGE_ORDER),
    *(f"nodal_{nodal.value}" for nodal in NODAL_STAGE_ORDER),
    "sex_male",
)

#: Width of the clinical input vector: 1 age + 5 stage + 5 nodal + 1 sex.
CLINICAL_FEATURE_DIM: Final[int] = len(CLINICAL_FEATURE_NAMES)

if CLINICAL_FEATURE_DIM != 12:
    raise ValueError(
        f"Clinical contract is 12-dimensional, built {CLINICAL_FEATURE_DIM}."
    )


class ClinicalData(BaseModel):
    """Structured clinical features for a single TCGA-BRCA patient.

    Every field is required. Unknown stage information is expressed through
    the ``UNKNOWN`` enum members, so there is no nullable state to handle
    downstream.
    """

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=0, le=120)
    sex: Sex
    tumor_stage: TumorStage
    nodal_stage: NodalStage
