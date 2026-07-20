"""replace the docstring."""

from datetime import date
from enum import Enum

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from src.data.schema.clinical import ClinicalData
from src.data.schema.imagery import ImageryData
from src.data.schema.genomics import GenomicsData


class CancerType(str, Enum):
    """replace the docstring."""

    BREAST = "breast"
    LUNG = "lung"
    PROSTATE = "prostate"


class Patient(BaseModel):
    """replace the docstring."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    cancer_type: CancerType
    date_of_diagnosis: Optional[date] = None

    clinical: Optional[ClinicalData] = None
    imagery: list[ImageryData] = Field(default_factory=list)
    genomics: list[GenomicsData] = Field(default_factory=list)

    def avaliable_modalities(self) -> list[str]:
        """replace the docstring."""

        mod: list[str] = []

        if self.clinical:
            mod.append("clinical")
        if self.imagery:
            mod.append("imagery")
        if self.genomics:
            mod.append("genomics")

        return mod
