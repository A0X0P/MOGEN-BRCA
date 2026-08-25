"""Unit tests for :mod:`src.data.preprocessing`.

Each preprocessor is tested for: (1) a minimal valid fixture that produces a
record validating against its schema, and (2) at least one malformed/edge
case (missing field, unmappable value, NaN, mismatched dims).

Clinical fixtures use raw token forms that actually occur in the TCGA-BRCA
source tables (``"STAGE IIA"``, ``"N1MI"``, ``"N0 (I-)"``, ``"STAGE X"``,
``"NX"``).
"""

from __future__ import annotations

import math

import pytest

from src.data.preprocessing.genomics_preprocess import GenomicsPreprocessor
from src.data.preprocessing.tabular_preprocess import ClinicalPreprocessor
from src.data.schema.clinical import (
    ClinicalData,
    NodalStage,
    Sex,
    TumorStage,
)
from src.data.schema.genomics import GenomicsData, OmicsType


# --------------------------------------------------------------------------- #
# Clinical
# --------------------------------------------------------------------------- #
def test_clinical_minimal_row_validates() -> None:
    raw = {
        "age": 61,
        "sex": "Female",
        "tumor_stage": "STAGE IIA",
        "nodal_stage": "N1A",
    }

    record = ClinicalPreprocessor().process(raw)

    assert isinstance(record, ClinicalData)
    assert record.age == 61
    assert record.sex is Sex.FEMALE
    assert record.tumor_stage is TumorStage.STAGE_II  # Substage collapsed.
    assert record.nodal_stage is NodalStage.N1


@pytest.mark.parametrize(
    ("raw_token", "expected"),
    [
        ("STAGE I", TumorStage.STAGE_I),
        ("STAGE IA", TumorStage.STAGE_I),
        ("STAGE IB", TumorStage.STAGE_I),
        ("STAGE II", TumorStage.STAGE_II),
        ("STAGE IIA", TumorStage.STAGE_II),
        ("STAGE IIB", TumorStage.STAGE_II),
        ("STAGE III", TumorStage.STAGE_III),
        ("STAGE IIIA", TumorStage.STAGE_III),
        ("STAGE IIIB", TumorStage.STAGE_III),
        ("STAGE IIIC", TumorStage.STAGE_III),
        ("STAGE IV", TumorStage.STAGE_IV),
        ("STAGE X", TumorStage.UNKNOWN),
        ("", TumorStage.UNKNOWN),
        (None, TumorStage.UNKNOWN),
    ],
)
def test_clinical_tumor_stage_collapsing(raw_token: object, expected: TumorStage) -> None:
    """Every tumour-stage token present in TCGA-BRCA collapses as ratified."""
    assert ClinicalPreprocessor.normalize_tumor_stage(raw_token) is expected


@pytest.mark.parametrize(
    ("raw_token", "expected"),
    [
        ("N0", NodalStage.N0),
        # Isolated tumour cells / molecular-only findings are staged N0.
        ("N0 (I-)", NodalStage.N0),
        ("N0 (I+)", NodalStage.N0),
        ("N0 (MOL+)", NodalStage.N0),
        ("N1", NodalStage.N1),
        ("N1A", NodalStage.N1),
        ("N1B", NodalStage.N1),
        ("N1C", NodalStage.N1),
        # Micrometastasis (pN1mi) is staged N1, not N0.
        ("N1MI", NodalStage.N1),
        ("N2", NodalStage.N2),
        ("N2A", NodalStage.N2),
        ("N3", NodalStage.N3),
        ("N3A", NodalStage.N3),
        ("N3B", NodalStage.N3),
        ("N3C", NodalStage.N3),
        ("NX", NodalStage.UNKNOWN),
        ("", NodalStage.UNKNOWN),
        (None, NodalStage.UNKNOWN),
    ],
)
def test_clinical_nodal_stage_collapsing(raw_token: object, expected: NodalStage) -> None:
    """Every nodal-stage token present in TCGA-BRCA collapses as ratified."""
    assert ClinicalPreprocessor.normalize_nodal_stage(raw_token) is expected


def test_clinical_missing_stage_becomes_unknown_not_error() -> None:
    """Absent stage is an explicit category, not a hard failure."""
    raw = {"age": 61, "sex": "Male"}

    record = ClinicalPreprocessor().process(raw)

    assert record.tumor_stage is TumorStage.UNKNOWN
    assert record.nodal_stage is NodalStage.UNKNOWN


def test_clinical_missing_required_field_raises() -> None:
    raw = {"sex": "Female", "tumor_stage": "STAGE I", "nodal_stage": "N0"}  # no age

    with pytest.raises(KeyError):
        ClinicalPreprocessor().process(raw)


def test_clinical_nan_required_field_raises() -> None:
    raw = {
        "age": float("nan"),
        "sex": "Female",
        "tumor_stage": "STAGE I",
        "nodal_stage": "N0",
    }

    with pytest.raises(KeyError):
        ClinicalPreprocessor().process(raw)


def test_clinical_unmappable_stage_raises() -> None:
    """An unrecognised token must fail loudly, not silently become Unknown."""
    with pytest.raises(ValueError):
        ClinicalPreprocessor.normalize_tumor_stage("banana")

    with pytest.raises(ValueError):
        ClinicalPreprocessor.normalize_nodal_stage("N7")


def test_clinical_unmappable_sex_raises() -> None:
    raw = {"age": 61, "sex": "unspecified", "tumor_stage": "STAGE I", "nodal_stage": "N0"}

    with pytest.raises(ValueError):
        ClinicalPreprocessor().process(raw)


def test_clinical_allow_list_excludes_outcome_columns() -> None:
    """The preprocessor reads only whitelisted feature columns."""
    assert ClinicalPreprocessor.ALLOWED_FIELDS == {
        "age",
        "sex",
        "tumor_stage",
        "nodal_stage",
    }


def test_clinical_ignores_extra_raw_columns() -> None:
    """Outcome columns present in a raw row never reach the schema."""
    raw = {
        "age": 61,
        "sex": "Female",
        "tumor_stage": "STAGE I",
        "nodal_stage": "N0",
        "OS_MONTHS": 42.0,
        "OS_STATUS": "1:DECEASED",
        "SUBTYPE": "BRCA_LumA",
    }

    record = ClinicalPreprocessor().process(raw)

    assert set(record.model_dump()) == {"age", "sex", "tumor_stage", "nodal_stage"}


# --------------------------------------------------------------------------- #
# Genomics
# --------------------------------------------------------------------------- #
def test_genomics_aligns_to_vocabulary() -> None:
    vocab = ["BRCA1", "TP53", "EGFR"]
    pre = GenomicsPreprocessor(OmicsType.CNV, vocab)

    record = pre.process({"TP53": 2.0, "BRCA1": -1.0})  # EGFR missing, unordered.

    assert isinstance(record, GenomicsData)
    assert record.feature_ids == vocab
    assert record.values == [-1.0, 2.0, 0.0]  # aligned + imputed with fill=0.


def test_genomics_log1p_applied_to_rna_seq() -> None:
    pre = GenomicsPreprocessor(OmicsType.RNA_SEQ, ["G1", "G2"])

    record = pre.process({"G1": 0.0, "G2": math.e - 1})

    assert record.values[0] == pytest.approx(0.0)
    assert record.values[1] == pytest.approx(1.0)


def test_genomics_nan_value_imputed_before_validator() -> None:
    pre = GenomicsPreprocessor(OmicsType.CNV, ["G1", "G2"])

    record = pre.process({"G1": float("nan"), "G2": 1.5})

    assert record.values == [0.0, 1.5]
    assert len(record.values) == len(record.feature_ids)


def test_genomics_negative_counts_rejected() -> None:
    pre = GenomicsPreprocessor(OmicsType.RNA_SEQ, ["G1"])

    with pytest.raises(ValueError):
        pre.process({"G1": -5.0})


def test_genomics_empty_vocabulary_rejected() -> None:
    with pytest.raises(ValueError):
        GenomicsPreprocessor(OmicsType.MIRNA, [])
