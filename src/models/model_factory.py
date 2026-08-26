"""Complete multimodal breast-cancer prediction model."""

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn

from src.data.pam50 import PAM50_GENE_COUNT
from src.data.schema.clinical import CLINICAL_FEATURE_DIM
from src.models.classification.multitask_head import (
    MultiTaskClassificationHead,
)
from src.models.fusion.cross_modal_attention import (
    CrossModalAttention,
)
from src.models.fusion.fusion_head import FusionHead
from src.models.genomics.genomic_transformer import (
    GenomicTransformer,
)
from src.models.survival.deepsurv_head import DeepSurvHead
from src.models.tabular.clinical_mlp import ClinicalMLP


@dataclass
class ModelConfig:
    """Configuration for the clinical-genomic breast-cancer model."""

    # Shared representation dimensions.
    embedding_dim: int = 256
    fused_dim: int = 256

    # General regularization.
    dropout: float = 0.1

    # Modalities.
    enable_clinical: bool = True
    enable_genomics: bool = True
    enable_survival: bool = True

    # Clinical encoder. The input width is the ratified 12-dimensional
    # clinical contract, not a free parameter.
    clinical_input_dim: int = CLINICAL_FEATURE_DIM
    clinical_hidden_dim: int = 128
    clinical_num_blocks: int = 2

    # Genomic Transformer. The input width is the 50-gene PAM50 panel.
    genomics_input_dim: int = PAM50_GENE_COUNT
    genomics_d_model: int = 128
    genomics_nhead: int = 4
    genomics_num_layers: int = 3
    genomics_dim_feedforward: int = 256

    # Cross-modal attention.
    cross_attention_nhead: int = 4
    cross_attention_dropout: float = 0.1

    # Multi-task classification.
    classification_hidden_dim: int = 128
    subtype_num_classes: int = 5
    receptor_num_classes: int = 2

    # Survival.
    survival_hidden_dim: int = 128


class MultimodalCancerModel(nn.Module):
    """Multimodal clinical-genomic breast-cancer prediction model.

    Architecture:

        Clinical data
             ↓
        ClinicalMLP
             ↓
        Clinical embedding
             │
             ├──── Cross-modal attention ────┐
             │                               │
             │                         Genomic embedding
             │                               ↑
             │                       Genomic Transformer
             │                               ↑
             │                      Gene-expression data
             │
             └──────────────┬────────────────┘
                            ↓
                    Concatenation Fusion
                            ↓
                      Fused representation
                            │
             ┌──────────────┼───────────────┐
             ↓              ↓               ↓
          PAM50       ER / PR / HER2     DeepSurv
           head           heads          head
             ↓              ↓               ↓
        subtype logits   receptor logits  risk score
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        # ------------------------------------------------------------------
        # Modality encoders
        # ------------------------------------------------------------------

        self.clinical_encoder: ClinicalMLP | None = None
        self.genomics_encoder: GenomicTransformer | None = None

        if config.enable_clinical:
            self.clinical_encoder = ClinicalMLP(
                input_dim=config.clinical_input_dim,
                embedding_dim=config.embedding_dim,
                hidden_dim=config.clinical_hidden_dim,
                num_blocks=config.clinical_num_blocks,
                dropout=config.dropout,
            )

        if config.enable_genomics:
            self.genomics_encoder = GenomicTransformer(
                input_dim=config.genomics_input_dim,
                embedding_dim=config.embedding_dim,
                d_model=config.genomics_d_model,
                nhead=config.genomics_nhead,
                num_layers=config.genomics_num_layers,
                dim_feedforward=config.genomics_dim_feedforward,
                dropout=config.dropout,
            )

        # ------------------------------------------------------------------
        # Cross-modal attention
        # ------------------------------------------------------------------
        # Attention across modality tokens is only meaningful with two
        # modalities present. In a single-modality ablation it is omitted
        # entirely rather than instantiated and left inert, so the reported
        # parameter count reflects the parameters that actually train.

        self.cross_attention: CrossModalAttention | None = None

        if config.enable_clinical and config.enable_genomics:
            self.cross_attention = CrossModalAttention(
                embedding_dim=config.embedding_dim,
                nhead=config.cross_attention_nhead,
                dropout=config.cross_attention_dropout,
                num_modalities=2,
            )

        # ------------------------------------------------------------------
        # Fusion
        # ------------------------------------------------------------------

        self.fusion_head = FusionHead(
            embedding_dim=config.embedding_dim,
            fused_dim=config.fused_dim,
            dropout=config.dropout,
        )

        # ------------------------------------------------------------------
        # Multi-task classification
        # ------------------------------------------------------------------

        self.classification_head = MultiTaskClassificationHead(
            input_dim=config.fused_dim,
            hidden_dim=config.classification_hidden_dim,
            subtype_num_classes=config.subtype_num_classes,
            receptor_num_classes=config.receptor_num_classes,
            dropout=config.dropout,
        )

        # ------------------------------------------------------------------
        # Survival
        # ------------------------------------------------------------------

        self.survival_head: DeepSurvHead | None = None

        if config.enable_survival:
            self.survival_head = DeepSurvHead(
                input_dim=config.fused_dim,
                hidden_dim=config.survival_hidden_dim,
                dropout=config.dropout,
            )

    @property
    def active_modalities(self) -> tuple[str, ...]:
        """Modality inputs this model instance accepts, in canonical order.

        A single-modality ablation accepts only its own modality: supplying a
        disabled modality's tensor raises rather than being ignored, so callers
        that assemble forward inputs must consult this property.

        Returns:
            Some subset of ``("clinical", "genomics")``, ordered as
            :data:`~src.models.fusion.cross_modal_attention.CrossModalAttention.MODALITY_ORDER`.
        """
        return tuple(
            name
            for name, enabled in (
                ("clinical", self.clinical_encoder is not None),
                ("genomics", self.genomics_encoder is not None),
            )
            if enabled
        )

    def forward(
        self,
        clinical: torch.Tensor | None = None,
        genomics: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the complete multimodal forward pass.

        Args:
            clinical:
                Clinical feature tensor with shape:

                    (batch_size, clinical_input_dim)

                or ``None``.

            genomics:
                Gene-expression tensor with shape:

                    (batch_size, genomics_input_dim)

                or ``None``.

        Returns:
            Dictionary containing:

                subtype_logits
                er_logits
                pr_logits
                her2_logits
                fused
                risk_score (if survival is enabled)

        Raises:
            ValueError:
                If no modality is supplied or a modality is enabled but
                its input is missing.
        """

        embeddings: dict[str, torch.Tensor | None] = {}

        # --------------------------------------------------------------
        # Clinical branch
        # --------------------------------------------------------------

        if clinical is not None:
            if self.clinical_encoder is None:
                raise ValueError(
                    "Clinical input was supplied, but the clinical encoder is disabled."
                )

            embeddings["clinical"] = self.clinical_encoder(clinical)

        # --------------------------------------------------------------
        # Genomic branch
        # --------------------------------------------------------------

        if genomics is not None:
            if self.genomics_encoder is None:
                raise ValueError(
                    "Genomic input was supplied, but the genomic encoder is disabled."
                )

            embeddings["genomics"] = self.genomics_encoder(genomics)

        if not embeddings:
            raise ValueError("At least one modality input must be provided.")

        # --------------------------------------------------------------
        # Cross-modal attention
        # --------------------------------------------------------------
        # Present only in the two-modality configuration. A single-modality
        # ablation passes its encoder output straight to fusion, which marks
        # the absent modality with a zero vector and a 0 presence indicator.

        if self.cross_attention is not None:
            attended = self.cross_attention(embeddings)
        else:
            attended = {
                name: embedding
                for name, embedding in embeddings.items()
                if embedding is not None
            }

        # --------------------------------------------------------------
        # Fusion
        # --------------------------------------------------------------

        fused = self.fusion_head(attended)

        # --------------------------------------------------------------
        # Multi-task classification
        # --------------------------------------------------------------

        classification_outputs = self.classification_head(fused)

        output: dict[str, torch.Tensor] = {
            **classification_outputs,
            "fused": fused,
        }

        # --------------------------------------------------------------
        # Survival
        # --------------------------------------------------------------

        if self.survival_head is not None:
            output["risk_score"] = self.survival_head(fused)

        return output


def build_model(
    config: dict[str, Any],
) -> MultimodalCancerModel:
    """Build the model from a configuration dictionary.

    Args:
        config:
            Dictionary containing keys corresponding to ModelConfig fields.

    Returns:
        Initialized MultimodalCancerModel.
    """
    valid_fields = {field.name for field in fields(ModelConfig)}

    model_config = ModelConfig(
        **{key: value for key, value in config.items() if key in valid_fields}
    )

    return MultimodalCancerModel(model_config)
