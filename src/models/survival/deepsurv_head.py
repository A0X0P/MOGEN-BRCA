"""DeepSurv prediction head for Cox proportional hazards survival analysis."""

import torch
import torch.nn as nn


class DeepSurvHead(nn.Module):
    """Predicts patient risk scores for Cox proportional hazards survival.

    Takes a fused embedding and produces a scalar risk score per patient.
    Higher risk scores indicate higher predicted hazard. The output is
    used with a Cox partial likelihood loss during training.

    Architecture:
        Fused embedding → FC → LayerNorm → GELU → Dropout
        → FC → LayerNorm → GELU → Dropout → Linear(1) → Risk score

    Args:
        input_dim: Dimensionality of the input fused embedding.
        hidden_dim: Width of hidden layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict risk score from fused embedding.

        Args:
            x: Tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, 1) containing risk scores.
        """
        return self.network(x)
