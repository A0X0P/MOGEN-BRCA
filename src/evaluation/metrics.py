"""Pure metric functions for classification, survival, and calibration.

Most functions take numpy arrays and return a scalar float. Two return
structured counts instead: :func:`confusion_matrix_counts` returns a matrix and
:func:`positive_class_report` returns a small mapping. None has side effects.
scikit-learn is used for classification metrics and Brier score.
Concordance index is implemented directly (lifelines is not a project dependency).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
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


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy: the unweighted mean of the per-class recalls.

    For a single-label task this is numerically identical to macro-averaged
    recall (:func:`recall` with ``average="macro"``). It is exposed separately
    because reporting conventions name the two differently, and because a
    reader should not have to know they coincide.

    Args:
        y_true: Ground-truth class indices, shape ``(N,)``.
        y_pred: Predicted class indices, shape ``(N,)``.

    Returns:
        Balanced accuracy in ``[0, 1]``. A model that predicts one class for
        every patient scores ``1 / n_classes``.
    """
    return float(balanced_accuracy_score(y_true, y_pred))


def confusion_matrix_counts(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> list[list[int]]:
    """Confusion matrix as nested integer lists, rows = true, columns = predicted.

    The class axis is pinned to ``range(n_classes)`` so the matrix keeps its
    full shape even when a partition contains no instances of a class, or the
    model never predicts one. A collapsed prediction is then visible as an
    all-zero column rather than as a silently smaller matrix.

    Args:
        y_true: Ground-truth class indices, shape ``(N,)``.
        y_pred: Predicted class indices, shape ``(N,)``.
        n_classes: Number of classes in the task's vocabulary.

    Returns:
        An ``n_classes x n_classes`` matrix of counts.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return [[int(value) for value in row] for row in matrix]


def positive_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1
) -> dict[str, float]:
    """Precision, recall, specificity and F1 for the positive class alone.

    Macro averages hide the behaviour of a minority class: a model that never
    predicts positive can still post a respectable macro-F1. For the imbalanced
    receptor tasks the positive class is reported on its own.

    Args:
        y_true: Ground-truth class indices, shape ``(N,)``.
        y_pred: Predicted class indices, shape ``(N,)``.
        pos_label: The class treated as positive.

    Returns:
        Mapping with ``precision``, ``recall`` (sensitivity), ``specificity``,
        ``f1``, ``support`` and ``n_predicted_positive``.
    """
    true_binary = (np.asarray(y_true) == pos_label).astype(int)
    pred_binary = (np.asarray(y_pred) == pos_label).astype(int)

    negatives = int((true_binary == 0).sum())
    true_negatives = int(((true_binary == 0) & (pred_binary == 0)).sum())

    return {
        "precision": float(
            precision_score(true_binary, pred_binary, zero_division=0)
        ),
        "recall": float(recall_score(true_binary, pred_binary, zero_division=0)),
        "specificity": float(true_negatives / negatives) if negatives else float("nan"),
        "f1": float(f1_score(true_binary, pred_binary, zero_division=0)),
        "support": float(true_binary.sum()),
        "n_predicted_positive": float(pred_binary.sum()),
    }


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
