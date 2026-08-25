"""Evaluation package: metrics, model evaluation, and interpretability."""

from src.evaluation.evaluator import (
    EvaluationResult,
    PredictionCollector,
    evaluate,
)
from src.evaluation.metrics import (
    accuracy,
    brier_score,
    concordance_index,
    f1,
    pr_auc,
    precision,
    recall,
    roc_auc,
)

__all__ = [
    "EvaluationResult",
    "PredictionCollector",
    "evaluate",
    "accuracy",
    "brier_score",
    "concordance_index",
    "f1",
    "pr_auc",
    "precision",
    "recall",
    "roc_auc",
]
