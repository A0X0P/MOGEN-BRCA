"""PyTorch Dataset for TCGA-BRCA clinical/tabular data.

Encodes validated :class:`~src.data.schema.clinical.ClinicalData` records into
the ratified 12-dimensional clinical vector:

    [age, stage I, II, III, IV, Unknown, nodal N0, N1, N2, N3, Unknown, sex_male]

The column order is defined once, in
:data:`~src.data.schema.clinical.CLINICAL_FEATURE_NAMES`, and this module
encodes against it. Age is the only continuous feature and is the only one
z-scored; the normalisation statistics must be fitted on the training
partition only (CLAUDE.md section 12).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from src.data.schema.clinical import (
    CLINICAL_FEATURE_DIM,
    CLINICAL_FEATURE_NAMES,
    NODAL_STAGE_ORDER,
    TUMOR_STAGE_ORDER,
    ClinicalData,
    Sex,
)

#: Continuous columns eligible for z-score normalisation.
_CONTINUOUS_FEATURES: tuple[str, ...] = ("age",)


def fit_normalization_stats(
    samples: list[ClinicalData],
) -> dict[str, dict[str, float]]:
    """Fit z-score statistics for the continuous clinical features.

    Call this with the TRAINING partition only, then pass the result to the
    validation and test datasets (CLAUDE.md section 12).

    Args:
        samples: Training-partition clinical records.

    Returns:
        Mapping of feature name -> ``{"mean": float, "std": float}``.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("Cannot fit normalization statistics on zero samples.")

    ages = torch.tensor([float(sample.age) for sample in samples])
    return {
        "age": {
            "mean": float(ages.mean()),
            # Population std: a single-sample fold must not produce NaN.
            "std": float(ages.std(unbiased=False)),
        }
    }


class TabularDataset(Dataset):
    """Converts validated ``ClinicalData`` objects into 12-dim feature tensors.

    Stage and nodal stage are one-hot encoded over the five-way vocabularies
    (including their explicit ``Unknown`` column), so missing stage
    information is represented rather than imputed. Sex is a single indicator
    column.
    """

    def __init__(
        self,
        samples: list[ClinicalData],
        labels: Optional[list[Any]] = None,
        normalization_stats: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        """Initialise the TabularDataset.

        Args:
            samples: List of validated ``ClinicalData`` schema objects.
            labels: Optional parallel list of labels.
            normalization_stats: Mapping of feature name to
                ``{"mean": float, "std": float}`` as produced by
                :func:`fit_normalization_stats`. When ``None``, continuous
                features are passed through unnormalised.

        Raises:
            ValueError: If ``labels`` is given with a mismatched length.
        """
        if labels is not None and len(labels) != len(samples):
            raise ValueError(
                f"Length mismatch: {len(samples)} samples vs {len(labels)} labels."
            )

        self._samples = samples
        self._labels = labels
        self._norm_stats = normalization_stats or {}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a feature tensor and optional label for one patient.

        Returns:
            Dict with key ``"features"`` (FloatTensor of shape ``(12,)``) and
            optionally ``"label"``.
        """
        clinical = self._samples[index]
        features = self.encode(clinical, self._norm_stats)

        result: dict[str, Any] = {"features": features}

        if self._labels is not None:
            label = self._labels[index]
            if isinstance(label, (int, float)):
                result["label"] = torch.tensor(label, dtype=torch.float32)
            else:
                result["label"] = label

        return result

    @staticmethod
    def encode(
        clinical: ClinicalData,
        normalization_stats: Optional[dict[str, dict[str, float]]] = None,
    ) -> torch.Tensor:
        """Encode one clinical record into the 12-dimensional feature vector.

        Args:
            clinical: Validated clinical record.
            normalization_stats: Optional z-score statistics for continuous
                features.

        Returns:
            FloatTensor of shape ``(12,)`` ordered as
            :data:`~src.data.schema.clinical.CLINICAL_FEATURE_NAMES`.

        Raises:
            ValueError: If the encoded width is not 12.
        """
        stats = normalization_stats or {}

        features: list[float] = [
            _normalize("age", float(clinical.age), stats),
            *(float(clinical.tumor_stage is stage) for stage in TUMOR_STAGE_ORDER),
            *(float(clinical.nodal_stage is nodal) for nodal in NODAL_STAGE_ORDER),
            float(clinical.sex is Sex.MALE),
        ]

        if len(features) != CLINICAL_FEATURE_DIM:
            raise ValueError(
                f"Clinical encoding produced {len(features)} features, "
                f"expected {CLINICAL_FEATURE_DIM}."
            )

        return torch.tensor(features, dtype=torch.float32)

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vector (always 12)."""
        return CLINICAL_FEATURE_DIM

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Column names of the output feature vector, in order."""
        return CLINICAL_FEATURE_NAMES


def _normalize(
    feature_name: str, value: float, stats: dict[str, dict[str, float]]
) -> float:
    """Apply z-score normalisation when statistics are available.

    Args:
        feature_name: Name of the feature being encoded.
        value: Raw feature value.
        stats: Mapping of feature name to ``{"mean": ..., "std": ...}``.

    Returns:
        The normalised value, or ``value`` unchanged when no usable statistics
        exist for this feature (including a zero-variance fold).
    """
    if feature_name not in _CONTINUOUS_FEATURES or feature_name not in stats:
        return value

    entry = stats[feature_name]
    std = entry["std"]
    if std <= 0:
        return value
    return (value - entry["mean"]) / std
