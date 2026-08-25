"""Pure metric functions for classification, survival, and calibration.

Each function takes numpy arrays and returns a scalar float. No side effects.
scikit-learn is used for classification metrics and Brier score.
Concordance index is implemented directly (lifelines is not a project dependency).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples.

    Args:
        y_true: Ground-truth class indices, shape ``(N,)``.
        y_pred: Predicted class indices, shape ``(N,)``.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    return float(accuracy_score(y_true, y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """Precision score.

    Args:
        y_true: Ground-truth class indices, shape ``(N,)``.
        y_pred: Predicted class indices, shape ``(N,)``.
        average: Averaging strategy (``"macro"``, ``"weighted"``, etc.).

    Returns:
        Precision in ``[0, 1]``.
    """
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))


def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """Recall score.

    Args:
        y_true: Ground-truth class indices, shape ``(N,)``.
        y_pred: Predicted class indices, shape ``(N,)``.
        average: Averaging strategy.

    Returns:
        Recall in ``[0, 1]``.
    """
    return float(recall_score(y_true, y_pred, average=average, zero_division=0))


def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """F1 score.

    Args:
        y_true: Ground-truth class indices, shape ``(N,)``.
        y_pred: Predicted class indices, shape ``(N,)``.
        average: Averaging strategy.

    Returns:
        F1 in ``[0, 1]``.
    """
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    multi_class: str = "ovr",
) -> float:
    """Area under the ROC curve.

    Args:
        y_true: Ground-truth class indices, shape ``(N,)``.
        y_prob: Class probabilities, shape ``(N, C)`` or ``(N,)`` for binary.
        multi_class: Strategy for multi-class (``"ovr"`` or ``"ovo"``).

    Returns:
        ROC-AUC in ``[0, 1]``.
    """
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    return float(roc_auc_score(y_true, y_prob, multi_class=multi_class))


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    Args:
        y_true: Ground-truth class indices, shape ``(N,)``.
        y_prob: Positive-class probabilities, shape ``(N,)`` for binary
            or ``(N, C)`` for multi-class (macro-averaged).

    Returns:
        PR-AUC in ``[0, 1]``.
    """
    if y_prob.ndim == 2:
        scores = [
            average_precision_score((y_true == c).astype(int), y_prob[:, c])
            for c in range(y_prob.shape[1])
        ]
        return float(np.mean(scores))
    return float(average_precision_score(y_true, y_prob))


def concordance_index(
    risk_scores: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
) -> float:
    """Harrell's concordance index (C-index) for survival models.

    Fraction of comparable pairs where the higher-risk patient experienced
    the event first. 0.5 is random; 1.0 is perfect.

    Args:
        risk_scores: Predicted log-risk scores, shape ``(N,)``.
        durations: Observed follow-up times, shape ``(N,)``.
        events: Event indicators (1=event, 0=censored), shape ``(N,)``.

    Returns:
        C-index in ``[0, 1]``, or 0.5 when no comparable pairs exist.
    """
    risk_scores = np.asarray(risk_scores, dtype=np.float64).ravel()
    durations = np.asarray(durations, dtype=np.float64).ravel()
    events = np.asarray(events, dtype=np.float64).ravel()

    concordant = 0.0
    comparable = 0

    for i, _ in enumerate(durations):
        if events[i] == 0:
            continue
        for j, _ in enumerate(durations):
            if i == j or durations[i] >= durations[j]:
                continue
            comparable += 1
            if risk_scores[i] > risk_scores[j]:
                concordant += 1.0
            elif risk_scores[i] == risk_scores[j]:
                concordant += 0.5

    return float(concordant / comparable) if comparable > 0 else 0.5


def brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pos_label: int = 1,
) -> float:
    """Brier score for calibration.

    Args:
        y_true: Ground-truth class indices, shape ``(N,)``.
        y_prob: Positive-class probability estimates, shape ``(N,)``.
        pos_label: The class treated as positive.

    Returns:
        Brier score in ``[0, 1]``; lower is better.
    """
    binary_true = (y_true == pos_label).astype(int)
    return float(brier_score_loss(binary_true, y_prob))
