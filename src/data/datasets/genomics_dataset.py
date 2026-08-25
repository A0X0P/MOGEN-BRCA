"""PyTorch Dataset for genomics/omics data."""

from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.schema.genomics import GenomicsData


class GenomicsDataset(Dataset):
    """Converts validated GenomicsData schema objects into feature tensors.

    Maps variable-length feature_ids onto a fixed vocabulary index shared
    across all samples, producing dense tensors with consistent ordering.
    Features not present in a given sample are zero-filled.
    """

    def __init__(
        self,
        samples: list[GenomicsData],
        labels: list[Any] | None = None,
        feature_vocabulary: list[str] | None = None,
    ) -> None:
        """Initialize the GenomicsDataset.

        Args:
            samples: List of validated GenomicsData schema objects.
            labels: Optional parallel list of labels.
            feature_vocabulary: Ordered list of all possible feature_ids
                defining the output tensor dimension and ordering. If None,
                the vocabulary is inferred from the union of all samples'
                feature_ids (sorted for determinism).
        """
        if labels is not None and len(labels) != len(samples):
            raise ValueError(
                f"Length mismatch: {len(samples)} samples vs {len(labels)} labels."
            )

        self._samples = samples
        self._labels = labels

        if feature_vocabulary is not None:
            self._vocabulary = feature_vocabulary
        else:
            self._vocabulary = self._build_vocabulary(samples)

        self._vocab_index: dict[str, int] = {
            fid: idx for idx, fid in enumerate(self._vocabulary)
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a dense feature tensor for one genomics sample.

        Returns:
            Dict with keys "omics_type" (str), "features" (FloatTensor),
            and optionally "label".
        """
        genomics = self._samples[index]
        features = self._to_dense_tensor(genomics)

        result: dict[str, Any] = {
            "omics_type": genomics.omics_type.value,
            "features": features,
        }

        if self._labels is not None:
            label = self._labels[index]
            if isinstance(label, (int, float)):
                result["label"] = torch.tensor(label, dtype=torch.float32)
            else:
                result["label"] = label

        return result

    def _to_dense_tensor(self, genomics: GenomicsData) -> torch.Tensor:
        """Map sparse feature_ids/values onto the fixed vocabulary index."""
        tensor = torch.zeros(len(self._vocabulary), dtype=torch.float32)

        for fid, value in zip(genomics.feature_ids, genomics.values):
            if fid in self._vocab_index:
                tensor[self._vocab_index[fid]] = value

        return tensor

    @staticmethod
    def _build_vocabulary(samples: list[GenomicsData]) -> list[str]:
        """Build a sorted vocabulary from the union of all feature_ids."""
        all_ids: set[str] = set()
        for sample in samples:
            all_ids.update(sample.feature_ids)
        return sorted(all_ids)

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output feature tensor."""
        return len(self._vocabulary)

    @property
    def vocabulary(self) -> list[str]:
        """The fixed feature vocabulary used for tensor construction."""
        return list(self._vocabulary)
