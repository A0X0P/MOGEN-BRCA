"""Preprocessing for raw genomics records into :class:`GenomicsData`.

Raw-source assumption (one per module): each patient's omics data arrives as a
mapping of ``feature_id -> value`` (e.g. a gene-expression row keyed by gene
symbol, as produced by ``pandas.Series.to_dict()`` on a TCGA expression
matrix). Values may be missing or NaN.

The critical responsibility here is *feature-vocabulary alignment*: models
expect a fixed, ordered feature vector per omics type. The preprocessor is
constructed with an explicit feature vocabulary and always emits values in
that order, imputing missing features deterministically.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.data.schema.genomics import GenomicsData, OmicsType
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Omics types whose raw values are non-negative counts and benefit from a
# log1p transform. Others (already-normalised ratios/beta-values) are left as-is.
_LOG1P_OMICS = {OmicsType.RNA_SEQ, OmicsType.MIRNA}


class GenomicsPreprocessor:
    """Aligns raw omics maps to a fixed vocabulary and builds ``GenomicsData``.

    Args:
        omics_type: The omics modality this preprocessor handles.
        feature_ids: The canonical, ordered feature vocabulary. Every output
            record uses exactly these ids in this order.
        source: Provenance tag stored on each record (default ``"TCGA"``).
        missing_fill: Value substituted for features absent from a patient's
            raw map or present as NaN. Applied *before* normalization.

    The preprocessor is deterministic: the same raw map and vocabulary always
    produce the same aligned, normalized vector.
    """

    def __init__(
        self,
        omics_type: OmicsType,
        feature_ids: Sequence[str],
        *,
        source: str = "TCGA",
        missing_fill: float = 0.0,
    ) -> None:
        if not feature_ids:
            raise ValueError("feature_ids vocabulary must be non-empty.")
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("feature_ids vocabulary contains duplicates.")

        self._omics_type = omics_type
        self._feature_ids = list(feature_ids)
        self._source = source
        self._missing_fill = float(missing_fill)

    @property
    def feature_ids(self) -> list[str]:
        """Return a copy of the canonical feature vocabulary."""
        return list(self._feature_ids)

    def process(self, raw: Mapping[str, Any]) -> GenomicsData:
        """Align, impute, and normalize a single patient's omics map.

        Args:
            raw: Mapping of ``feature_id -> value`` for one patient.

        Returns:
            A validated :class:`GenomicsData` record whose ``feature_ids`` and
            ``values`` are aligned to the canonical vocabulary.
        """
        aligned = self._align(raw)
        normalized = self._normalize(aligned)

        return GenomicsData(
            omics_type=self._omics_type,
            feature_ids=self.feature_ids,
            values=normalized.tolist(),
            source=self._source,
        )

    def _align(self, raw: Mapping[str, Any]) -> np.ndarray:
        """Project a raw map onto the canonical vocabulary, imputing gaps.

        Missing and NaN features are filled with ``missing_fill``. Features in
        ``raw`` that are outside the vocabulary are dropped (and logged), since
        models cannot consume an unknown-width vector.
        """
        values = np.full(len(self._feature_ids), self._missing_fill, dtype=np.float64)

        missing = 0
        for index, feature_id in enumerate(self._feature_ids):
            if feature_id not in raw:
                missing += 1
                continue
            value = raw[feature_id]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                missing += 1
                continue
            values[index] = float(value)

        if missing:
            logger.warning(
                "%s: imputed %d/%d missing features with fill=%s.",
                self._omics_type.value,
                missing,
                len(self._feature_ids),
                self._missing_fill,
            )

        extra = set(raw) - set(self._feature_ids)
        if extra:
            logger.warning(
                "%s: dropped %d features outside the vocabulary.",
                self._omics_type.value,
                len(extra),
            )

        return values

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Apply per-omics normalization.

        ``rna_seq``/``mirna`` counts get a ``log1p`` transform; other omics
        types are returned unchanged. Negative counts are rejected loudly
        rather than silently producing NaNs.
        """
        if self._omics_type not in _LOG1P_OMICS:
            return values

        if np.any(values < 0):
            raise ValueError(
                f"{self._omics_type.value}: negative values are invalid for a "
                "count-based omics type and cannot be log-transformed."
            )
        return np.log1p(values)
