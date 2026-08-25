"""Concatenation-based fusion head for clinical and genomic representations."""

import torch
from torch import nn


class FusionHead(nn.Module):
    """Fuse attention-enhanced clinical and genomic representations.

    The fusion strategy is:

        Clinical representation
                    \\
                     → Concatenation → Fusion MLP → fused representation
                    /
        Genomic representation

    Missing modalities are represented by zero vectors and accompanied by
    binary modality-presence indicators.

    Args:
        embedding_dim: Dimensionality of each modality embedding.
        fused_dim: Dimensionality of the final fused representation.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        fused_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self._embedding_dim = embedding_dim
        self._fused_dim = fused_dim

        # Two modality embeddings plus two modality-presence indicators.
        fusion_input_dim = (embedding_dim * 2) + 2

        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(fused_dim)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of each modality embedding."""
        return self._embedding_dim

    @property
    def fused_dim(self) -> int:
        """Dimensionality of the fused representation."""
        return self._fused_dim

    def forward(
        self,
        modality_embeddings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fuse clinical and genomic embeddings.

        Args:
            modality_embeddings:
                Dictionary containing:

                    "clinical": (batch, embedding_dim)
                    "genomics": (batch, embedding_dim)

                Either modality may be absent.

        Returns:
            Tensor with shape:

                (batch, fused_dim)

        Raises:
            ValueError:
                If no modality embeddings are provided or an unsupported
                modality is supplied.
        """

        if not modality_embeddings:
            raise ValueError("At least one modality embedding is required.")

        unsupported = set(modality_embeddings) - {"clinical", "genomics"}

        if unsupported:
            raise ValueError(f"Unsupported modalities: {sorted(unsupported)}.")

        clinical = modality_embeddings.get("clinical")
        genomics = modality_embeddings.get("genomics")

        reference = clinical if clinical is not None else genomics

        if reference is None:
            raise ValueError("At least one clinical or genomic embedding is required.")

        batch_size = reference.shape[0]
        device = reference.device
        dtype = reference.dtype

        if clinical is None:
            clinical = torch.zeros(
                batch_size,
                self.embedding_dim,
                device=device,
                dtype=dtype,
            )
            clinical_present = torch.zeros(
                batch_size,
                1,
                device=device,
                dtype=dtype,
            )
        else:
            clinical_present = torch.ones(
                batch_size,
                1,
                device=clinical.device,
                dtype=clinical.dtype,
            )

        if genomics is None:
            genomics = torch.zeros(
                batch_size,
                self.embedding_dim,
                device=device,
                dtype=dtype,
            )
            genomics_present = torch.zeros(
                batch_size,
                1,
                device=device,
                dtype=dtype,
            )
        else:
            genomics_present = torch.ones(
                batch_size,
                1,
                device=genomics.device,
                dtype=genomics.dtype,
            )

        fused_input = torch.cat(
            [
                clinical,
                genomics,
                clinical_present,
                genomics_present,
            ],
            dim=-1,
        )

        fused = self.fusion_mlp(fused_input)

        return self.output_norm(fused)
