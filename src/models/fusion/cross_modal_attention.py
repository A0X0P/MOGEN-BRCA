"""Cross-modal attention module for clinical and genomic representations."""

import torch
from torch import nn


class CrossModalAttention(nn.Module):
    """Learn interactions between clinical and genomic modality embeddings.

    Each available modality is represented as a token. Multi-head attention
    allows the modality tokens to exchange information before fusion.

    For the current breast-cancer model, the supported modalities are:

        clinical
        genomics

    Args:
        embedding_dim: Dimensionality of each modality embedding.
        nhead: Number of attention heads.
        dropout: Dropout probability in the attention and feed-forward layers.
        num_modalities: Number of supported modalities.
    """

    MODALITY_ORDER = ("clinical", "genomics")

    def __init__(
        self,
        embedding_dim: int = 256,
        nhead: int = 4,
        dropout: float = 0.1,
        num_modalities: int = 2,
    ) -> None:
        super().__init__()

        if num_modalities != len(self.MODALITY_ORDER):
            raise ValueError(
                f"num_modalities must be {len(self.MODALITY_ORDER)} "
                f"for the clinical + genomics architecture."
            )

        if embedding_dim % nhead != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by nhead ({nhead})."
            )

        self._embedding_dim = embedding_dim

        self.modality_embeddings = nn.Embedding(
            num_embeddings=num_modalities,
            embedding_dim=embedding_dim,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the modality embeddings."""
        return self._embedding_dim

    def forward(
        self,
        modality_embeddings: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor]:
        """Apply attention across the available modality embeddings.

        Args:
            modality_embeddings:
                Dictionary containing clinical and/or genomic embeddings.
                Each tensor must have shape:

                    (batch_size, embedding_dim)

                Missing modalities should be represented by ``None`` or
                omitted from the dictionary.

        Returns:
            Dictionary containing attention-enhanced embeddings for all
            available modalities.

        Raises:
            ValueError:
                If an unsupported modality is supplied or if embeddings
                have incompatible shapes.
        """

        available: list[tuple[str, int, torch.Tensor]] = []

        for modality_id, name in enumerate(self.MODALITY_ORDER):
            embedding = modality_embeddings.get(name)

            if embedding is not None:
                if embedding.ndim != 2:
                    raise ValueError(
                        f"{name} embedding must have shape "
                        f"(batch_size, embedding_dim), got {tuple(embedding.shape)}."
                    )

                if embedding.shape[-1] != self.embedding_dim:
                    raise ValueError(
                        f"{name} embedding has dimension {embedding.shape[-1]}, "
                        f"expected {self.embedding_dim}."
                    )

                available.append((name, modality_id, embedding))

        unsupported = set(modality_embeddings) - set(self.MODALITY_ORDER)
        if unsupported:
            raise ValueError(
                f"Unsupported modalities: {sorted(unsupported)}. "
                f"Supported modalities: {self.MODALITY_ORDER}."
            )

        if not available:
            raise ValueError("At least one modality embedding is required.")

        # With only one available modality, attention cannot provide
        # cross-modal interaction. Return the representation unchanged.
        if len(available) == 1:
            name, _, embedding = available[0]
            return {name: embedding}

        batch_size = available[0][2].shape[0]
        device = available[0][2].device

        for _, _, embedding in available:
            if embedding.shape[0] != batch_size:
                raise ValueError(
                    "All modality embeddings must have the same batch size."
                )

        # Shape:
        #     (batch, num_modalities, embedding_dim)
        tokens = torch.stack(
            [embedding for _, _, embedding in available],
            dim=1,
        )

        modality_ids = torch.tensor(
            [modality_id for _, modality_id, _ in available],
            dtype=torch.long,
            device=device,
        )

        modality_tokens = self.modality_embeddings(modality_ids).unsqueeze(0)

        tokens = tokens + modality_tokens

        # Modality-token attention.
        attended_tokens, _ = self.cross_attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )

        attended = self.norm1(tokens + attended_tokens)

        attended = self.norm2(attended + self.feed_forward(attended))

        result: dict[str, torch.Tensor] = {}

        for index, (name, _, _) in enumerate(available):
            result[name] = attended[:, index, :]

        return result
