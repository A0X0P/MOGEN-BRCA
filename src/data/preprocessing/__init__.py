"""Per-modality preprocessing that builds validated schema records.

Turns raw per-modality data into cleaned instances of the ``src.data.schema``
contracts, ready for consumption by ``src.data.datasets``.
"""

from src.data.preprocessing.genomics_preprocess import GenomicsPreprocessor
from src.data.preprocessing.tabular_preprocess import ClinicalPreprocessor

__all__ = [
    "ClinicalPreprocessor",
    "GenomicsPreprocessor",
]
