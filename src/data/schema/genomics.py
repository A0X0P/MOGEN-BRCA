"""Genomics data schema for a patient record.

Defines a single omics measurement (one entry per omics type per patient),
validated for structural consistency between feature identifiers and their
values. Normalization, log-transforms, and feature alignment across
patients belong in src/data/preprocessing/, not here.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator


class OmicsType(str, Enum):
    """Supported genomic/omics data types."""

    RNA_SEQ = "rna_seq"
    CNV = "cnv"
    METHYLATION = "methylation"
    MUTATION = "mutation"
    MIRNA = "mirna"


class GenomicsData(BaseModel):
    """A single omics record for a patient.

    Represents one omics measurement (e.g. one RNA-seq profile) as a
    parallel list of feature identifiers and their corresponding values.
    A patient may have multiple GenomicsData entries, one per omics type.
    """

    model_config = ConfigDict(extra="forbid")

    omics_type: OmicsType
    feature_ids: list[str]
    values: list[float]
    source: str = "TCGA"

    @field_validator("values")
    @classmethod
    def check_alignment(cls, v: list[float], info) -> list[float]:
        """Ensure values and feature_ids have matching length.

        Raises:
            ValueError: if the number of values does not match the number
                of feature identifiers.
        """

        feature_ids = info.data.get("feature_ids")
        if len(v) != len(feature_ids):
            raise ValueError("Values and Feature ids length mismatch.")
        return v
