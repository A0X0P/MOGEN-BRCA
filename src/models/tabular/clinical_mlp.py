"""Clinical tabular data encoder using residual MLP blocks."""

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    """Single residual MLP block: FC → LayerNorm → GELU → Dropout → FC → Add."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual MLP block.

        Args:
            x: Tensor of shape (batch, dim).

        Returns:
            Tensor of shape (batch, dim).
        """
        return x + self.net(x)


class ClinicalMLP(nn.Module):
    """Encodes clinical/tabular features into a compact latent representation.

    Architecture:
        Input → Linear projection → LayerNorm → GELU → Dropout
        → N residual MLP blocks → Clinical embedding

    Args:
        input_dim: Dimensionality of the input feature vector.
        embedding_dim: Dimensionality of the output embedding.
        hidden_dim: Width of hidden layers. Defaults to embedding_dim.
        num_blocks: Number of residual MLP blocks.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int | None = None,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""
        return self._embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode clinical features into an embedding.

        Args:
            x: Tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, embedding_dim).
        """
        h = self.input_projection(x)
        h = self.blocks(h)
        return self.output_projection(h)
