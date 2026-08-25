"""Dataset classes for the TCGA-BRCA two-modality framework.

Provides PyTorch Dataset implementations that convert validated schema
objects into tensors ready for model training and inference.
"""

from src.data.datasets.genomics_dataset import GenomicsDataset
from src.data.datasets.multimodal_dataset import (
    MultimodalDataset,
    extract_gene_vector,
    fit_gene_standardization,
)
from src.data.datasets.tabular_dataset import TabularDataset, fit_normalization_stats

__all__ = [
    "GenomicsDataset",
    "MultimodalDataset",
    "TabularDataset",
    "extract_gene_vector",
    "fit_gene_standardization",
    "fit_normalization_stats",
]
