"""Multi-task classification heads for breast-cancer prediction."""

import torch
from torch import nn


class ClassificationBranch(nn.Module):
    """Small MLP classification branch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce classification logits."""
        return self.network(x)


class MultiTaskClassificationHead(nn.Module):
    """Joint classification head for molecular subtype and receptor status.

    Tasks:

        PAM50 subtype:
            Luminal A
            Luminal B
            HER2-enriched
            Basal-like
            Normal-like

        ER status:
            Binary classification

        PR status:
            Binary classification

        HER2 status:
            Binary classification

    All tasks operate on the same fused multimodal representation.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        subtype_num_classes: int = 5,
        receptor_num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.subtype_head = ClassificationBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=subtype_num_classes,
            dropout=dropout,
        )

        self.er_head = ClassificationBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=receptor_num_classes,
            dropout=dropout,
        )

        self.pr_head = ClassificationBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=receptor_num_classes,
            dropout=dropout,
        )

        self.her2_head = ClassificationBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=receptor_num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate logits for all classification tasks.

        Args:
            x:
                Fused representation with shape:

                    (batch_size, input_dim)

        Returns:
            Dictionary containing logits for each prediction task.
        """

        return {
            "subtype_logits": self.subtype_head(x),
            "er_logits": self.er_head(x),
            "pr_logits": self.pr_head(x),
            "her2_logits": self.her2_head(x),
        }
