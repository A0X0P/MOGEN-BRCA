"""Transformer encoder for PAM50 gene-expression features."""

import torch
from torch import nn


class GenomicTransformer(nn.Module):
    """Transformer-based encoder for gene-expression features.

    Each gene-expression value is transformed into a token representation
    and combined with a learned gene-identity embedding. Transformer
    self-attention then models relationships between genes.

    Architecture:

        Gene expression values
                ↓
        Scalar feature projection
                +
        Learned gene identity embedding
                ↓
        Transformer encoder
                ↓
        Mean pooling
                ↓
        LayerNorm
                ↓
        Projection MLP
                ↓
        Genomic embedding

    Args:
        input_dim:
            Number of genomic features/genes supplied to the model.

        embedding_dim:
            Dimensionality of the final genomic representation.

        d_model:
            Internal Transformer representation dimension.

        nhead:
            Number of self-attention heads.

        num_layers:
            Number of Transformer encoder layers.

        dim_feedforward:
            Hidden dimension of the Transformer feed-forward network.

        dropout:
            Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )

        self._embedding_dim = embedding_dim
        self._input_dim = input_dim
        self._d_model = d_model

        # Projects each scalar expression value into the Transformer space.
        self.feature_embedding = nn.Linear(1, d_model)

        # Explicitly represents gene identity.
        #
        # The i-th embedding corresponds to the i-th gene in the fixed
        # feature ordering used by the preprocessing pipeline.
        self.gene_embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.projection = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output genomic embedding."""
        return self._embedding_dim

    @property
    def input_dim(self) -> int:
        """Number of genomic input features."""
        return self._input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode gene-expression features.

        Args:
            x:
                Gene-expression tensor with shape:

                    (batch_size, input_dim)

                Values should already have undergone the preprocessing
                specified by the data pipeline.

        Returns:
            Genomic embedding with shape:

                (batch_size, embedding_dim)
        """

        if x.ndim != 2:
            raise ValueError(
                f"Expected genomic input with shape "
                f"(batch_size, input_dim), got {tuple(x.shape)}."
            )

        _, num_features = x.shape  # _ = batch_size

        if num_features != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} genomic features, received {num_features}."
            )

        # Convert each scalar gene-expression value into a token.
        #
        # (B, G)
        #   ↓ unsqueeze
        # (B, G, 1)
        #   ↓ Linear
        # (B, G, d_model)
        tokens = self.feature_embedding(x.unsqueeze(-1))

        # Explicit gene identities.
        gene_ids = torch.arange(
            num_features,
            device=x.device,
            dtype=torch.long,
        )

        gene_embeddings = self.gene_embedding(gene_ids)

        # Broadcast gene identity embeddings across the batch.
        tokens = tokens + gene_embeddings.unsqueeze(0)

        # No expression-value-based padding mask is used.
        #
        # Zero can be a legitimate normalized expression value and therefore
        # should not automatically be interpreted as "missing".
        encoded = self.transformer(tokens)

        # Mean pooling across genes.
        pooled = encoded.mean(dim=1)

        pooled = self.layer_norm(pooled)

        return self.projection(pooled)
