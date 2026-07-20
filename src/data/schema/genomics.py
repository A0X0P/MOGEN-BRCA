"""replace the docstring."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator


class OmicsType(str, Enum):
    """replace the docstring."""

    RNA_SEQ = "rna_seq"
    CNV = "cnv"
    METHYLATION = "methylation"
    MUTATION = "mutation"
    MIRNA = "mirna"


class GenomicsData(BaseModel):
    """replace the docstring."""

    model_config = ConfigDict(extra="forbid")

    omics_type: OmicsType
    feature_ids: list[str]
    values: list[float]
    source: str = "TCGA"

    @field_validator("values")
    @classmethod
    def check_alignment(cls, v: list[float], info) -> list[float]:
        """replace the docstring."""

        feature_ids = info.data.get("feature_ids")
        if len(v) != len(feature_ids):
            raise ValueError("Values and Feature ids length mismatch.")
        return v
