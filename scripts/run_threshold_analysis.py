"""Select receptor decision thresholds on validation predictions, then apply once to test.

Usage:
    uv run scripts/run_threshold_analysis.py

Why this exists
---------------
At the default 0.5 threshold the frozen model detects only 6 of 32 HER2-positive
test patients (positive-class recall 0.1875). That is an operating-point choice,
not a ranking failure: ROC-AUC and average precision are threshold-free and
cannot be improved by moving the threshold. This script asks whether a different
operating point on the *same* frozen predictions trades specificity for
sensitivity usefully.

Protocol
--------
The selection rule is fixed in advance and uses **validation predictions only**:

1. Sweep thresholds 0.01 … 0.99 in 0.01 steps over the validation partition.
2. **Primary rule:** pick the threshold maximising validation balanced accuracy.
   Ties break toward the threshold nearest 0.5, then toward the smaller value.
3. **Secondary rule:** the same, maximising validation positive-class F1. Both
   are chosen before any test label is touched, and both are applied once, so
   neither is a post-hoc pick between two test outcomes.
4. A task's selected threshold is applied to test if it is HER2 (which motivated
   the analysis) or if its validation balanced-accuracy gain over 0.5 is at
   least ``MIN_VALIDATION_GAIN``. That criterion reads validation numbers only.

Test labels are not loaded until every threshold has been selected and written
into the result payload; :func:`main` enforces that ordering, and the payload
records the selected thresholds independently of the test outcome so the two can
be audited separately.

Nothing is retrained and no checkpoint is modified. Writes
``results/breast_analysis/threshold_analysis.json`` and ``threshold_sweep.csv``.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_detailed_metrics import RUNS, load_predictions  # noqa: E402
from src.data.tasks import RECEPTOR_TASKS  # noqa: E402
from src.evaluation import metrics  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

OUTPUT_DIR = REPO_ROOT / "results/breast_analysis"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

#: Candidate thresholds. A fixed 0.01 grid rather than the observed validation
#: probabilities, so the candidate set does not itself depend on the data.
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.01 * step, 2) for step in range(1, 100))

#: Default threshold the committed metrics were produced at.
DEFAULT_THRESHOLD = 0.5

#: Minimum validation balanced-accuracy gain over the default threshold required
#: before a non-HER2 task's selected threshold is applied to the test partition.
#: Declared here, and evaluated against validation numbers only.
MIN_VALIDATION_GAIN = 0.02


@dataclass(frozen=True)
class Operating:
    """Confusion counts and derived rates at one threshold.

    Attributes:
        threshold: Decision threshold applied to the positive-class probability.
        tp: True positives.
        fp: False positives.
        tn: True negatives.
        fn: False negatives.
    """

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def sensitivity(self) -> float:
        """Positive-class recall; 0.0 when the partition has no positives."""
        positives = self.tp + self.fn
        return self.tp / positives if positives else 0.0

    @property
    def specificity(self) -> float:
        """True-negative rate; 0.0 when the partition has no negatives."""
        negatives = self.tn + self.fp
        return self.tn / negatives if negatives else 0.0

    @property
    def precision(self) -> float:
        """Positive-class precision; 0.0 when nothing is predicted positive."""
        predicted = self.tp + self.fp
        return self.tp / predicted if predicted else 0.0

    @property
    def positive_f1(self) -> float:
        """Harmonic mean of positive-class precision and recall."""
        denominator = self.precision + self.sensitivity
        if denominator == 0.0:
            return 0.0
        return 2.0 * self.precision * self.sensitivity / denominator

    @property
    def balanced_accuracy(self) -> float:
        """Mean of sensitivity and specificity."""
        return 0.5 * (self.sensitivity + self.specificity)

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Flat mapping for JSON and CSV output."""
        return {
            "threshold": self.threshold,
            "n": self.n,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "positive_precision": self.precision,
            "positive_f1": self.positive_f1,
            "n_predicted_positive": self.tp + self.fp,
            "confusion_matrix": [[self.tn, self.fp], [self.fn, self.tp]],
        }


def task_arrays(rows: list[dict[str, str]], task: str) -> tuple[np.ndarray, np.ndarray]:
    """Labels and positive-class probabilities for one task's masked-in rows.

    Args:
        rows: Prediction rows read from an export CSV.
        task: Receptor task name.

    Returns:
        ``(y_true, prob_pos)`` over the patients whose label for ``task`` exists.
    """
    scored = [row for row in rows if row[f"{task}_mask"] == "1"]
    y_true = np.array([int(row[f"{task}_true"]) for row in scored])
    prob_pos = np.array([float(row[f"{task}_prob_pos"]) for row in scored])
    return y_true, prob_pos


def operating_point(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float) -> Operating:
    """Confusion counts obtained by predicting positive when ``prob >= threshold``."""
    predicted = prob_pos >= threshold
    positive = y_true == 1
    return Operating(
        threshold=threshold,
        tp=int((predicted & positive).sum()),
        fp=int((predicted & ~positive).sum()),
        tn=int((~predicted & ~positive).sum()),
        fn=int((~predicted & positive).sum()),
    )


def sweep(y_true: np.ndarray, prob_pos: np.ndarray) -> list[Operating]:
    """Operating points across the whole candidate grid."""
    return [operating_point(y_true, prob_pos, threshold) for threshold in THRESHOLD_GRID]


def _select(points: list[Operating], key: str) -> Operating:
    """Best operating point under ``key``, ties broken toward 0.5 then downward.

    Args:
        points: Validation operating points.
        key: ``Operating`` property name to maximise.

    Returns:
        The chosen operating point.
    """
    best = max(getattr(point, key) for point in points)
    tied = [point for point in points if getattr(point, key) == best]
    return min(tied, key=lambda point: (abs(point.threshold - DEFAULT_THRESHOLD), point.threshold))


def select_thresholds(validation_rows: list[dict[str, str]], task: str) -> dict[str, Any]:
    """Run the validation sweep and apply the pre-declared selection rules.

    Args:
        validation_rows: Validation prediction rows.
        task: Receptor task name.

    Returns:
        The sweep, the default operating point, both selected thresholds, and the
        pre-declared decision on whether to apply the primary threshold to test.
    """
    y_true, prob_pos = task_arrays(validation_rows, task)
    points = sweep(y_true, prob_pos)

    default = operating_point(y_true, prob_pos, DEFAULT_THRESHOLD)
    primary = _select(points, "balanced_accuracy")
    secondary = _select(points, "positive_f1")

    gain = primary.balanced_accuracy - default.balanced_accuracy
    mandated = task == "her2"
    apply_to_test = mandated or gain >= MIN_VALIDATION_GAIN

    return {
        "task": task,
        "partition_used_for_selection": "val",
        "n_validation_scored": int(len(y_true)),
        "n_validation_positive": int((y_true == 1).sum()),
        "n_validation_negative": int((y_true == 0).sum()),
        "validation_at_default_threshold": default.as_dict(),
        "primary_rule": "maximise validation balanced accuracy",
        "primary_threshold": primary.threshold,
        "validation_at_primary_threshold": primary.as_dict(),
        "validation_balanced_accuracy_gain": gain,
        "secondary_rule": "maximise validation positive-class F1",
        "secondary_threshold": secondary.threshold,
        "validation_at_secondary_threshold": secondary.as_dict(),
        "applied_to_test": apply_to_test,
        "application_reason": (
            "HER2 motivated the analysis and is applied unconditionally"
            if mandated
            else (
                f"validation balanced-accuracy gain {gain:.4f} "
                f"{'>=' if apply_to_test else '<'} {MIN_VALIDATION_GAIN} threshold"
            )
        ),
        "validation_sweep": [point.as_dict() for point in points],
    }


def apply_to_test(
    test_rows: list[dict[str, str]], task: str, selection: dict[str, Any]
) -> dict[str, Any]:
    """Apply the already-selected thresholds to the test partition exactly once.

    Args:
        test_rows: Test prediction rows.
        task: Receptor task name.
        selection: Output of :func:`select_thresholds` for the same task.

    Returns:
        Test operating points at the default and both selected thresholds, with
        the change in each reported quantity.
    """
    y_true, prob_pos = task_arrays(test_rows, task)

    default = operating_point(y_true, prob_pos, DEFAULT_THRESHOLD)
    primary = operating_point(y_true, prob_pos, selection["primary_threshold"])
    secondary = operating_point(y_true, prob_pos, selection["secondary_threshold"])

    changed = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "positive_f1")
    return {
        "n_test_scored": int(len(y_true)),
        "n_test_positive": int((y_true == 1).sum()),
        "n_test_negative": int((y_true == 0).sum()),
        "test_at_default_threshold": default.as_dict(),
        "test_at_primary_threshold": primary.as_dict(),
        "test_at_secondary_threshold": secondary.as_dict(),
        "primary_change_vs_default": {
            key: getattr(primary, key) - getattr(default, key) for key in changed
        },
        "secondary_change_vs_default": {
            key: getattr(secondary, key) - getattr(default, key) for key in changed
        },
        "threshold_free_metrics_unchanged": {
            "roc_auc": metrics.roc_auc(y_true, prob_pos),
            "average_precision_positive_class": metrics.pr_auc(y_true, prob_pos),
            "note": (
                "ROC-AUC and average precision rank patients and do not depend "
                "on the threshold. They are identical at every threshold above; "
                "moving the threshold only chooses a point on the same curve."
            ),
        },
    }


def git_commit() -> Optional[str]:
    """Current HEAD commit, or ``None`` outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sweep_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every validation sweep into CSV rows."""
    rows: list[dict[str, Any]] = []
    for run_name, run in results.items():
        for task, block in run["tasks"].items():
            for point in block["selection"]["validation_sweep"]:
                row = {"run": run_name, "task": task, "partition": "val"}
                row.update({key: value for key, value in point.items() if key != "confusion_matrix"})
                row["is_primary_selected"] = point["threshold"] == block["selection"]["primary_threshold"]
                row["is_secondary_selected"] = (
                    point["threshold"] == block["selection"]["secondary_threshold"]
                )
                rows.append(row)
    return rows


def main() -> None:
    """Select thresholds on validation, then evaluate them once on test."""
    ensure_dir(OUTPUT_DIR)

    # Phase 1: selection. Only validation predictions are read here.
    selections: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in RUNS:
        validation_rows = load_predictions(PREDICTIONS_DIR / f"{spec.name}_val.csv")
        selections[spec.name] = {
            task: select_thresholds(validation_rows, task) for task in RECEPTOR_TASKS
        }
        for task, selection in selections[spec.name].items():
            logger.info(
                "%s/%s: validation balanced accuracy %.4f at 0.50 -> %.4f at %.2f "
                "(gain %+.4f); applied to test: %s.",
                spec.name,
                task,
                selection["validation_at_default_threshold"]["balanced_accuracy"],
                selection["validation_at_primary_threshold"]["balanced_accuracy"],
                selection["primary_threshold"],
                selection["validation_balanced_accuracy_gain"],
                selection["applied_to_test"],
            )

    # Phase 2: application. Test predictions are read only now, after every
    # threshold above is fixed.
    results: dict[str, Any] = {}
    for spec in RUNS:
        test_rows = load_predictions(PREDICTIONS_DIR / f"{spec.name}_test.csv")
        tasks: dict[str, Any] = {}
        for task in RECEPTOR_TASKS:
            selection = selections[spec.name][task]
            block: dict[str, Any] = {"selection": selection}
            if selection["applied_to_test"]:
                block["test"] = apply_to_test(test_rows, task, selection)
            else:
                block["test"] = {
                    "skipped": (
                        "The pre-declared application criterion was not met on "
                        "validation, so the test partition was not re-scored at "
                        "a new threshold for this task."
                    )
                }
            tasks[task] = block
        results[spec.name] = {"label": spec.label, "frozen_reference": spec.frozen, "tasks": tasks}

    payload = {
        "purpose": (
            "Receptor decision-threshold selection on validation predictions, "
            "applied once to the test partition. Motivated by HER2 "
            "positive-class recall at the default 0.5 threshold."
        ),
        "analysis_git_commit": git_commit(),
        "predictions_source": "scripts/export_predictions.py",
        "protocol": {
            "candidate_grid": "0.01 to 0.99 in steps of 0.01",
            "selection_partition": "validation only; test labels not used for selection",
            "primary_rule": "maximise validation balanced accuracy",
            "secondary_rule": "maximise validation positive-class F1",
            "tie_break": "threshold nearest 0.50, then the smaller threshold",
            "application_criterion": (
                "HER2 unconditionally; any other receptor task only if its "
                f"validation balanced-accuracy gain is at least {MIN_VALIDATION_GAIN}"
            ),
            "applications_per_task": 1,
            "retraining": "none; the frozen checkpoints were not modified",
        },
        "caveats": {
            "threshold_free_metrics": (
                "Thresholding cannot change ROC-AUC or average precision. Any "
                "improvement below is a movement along the existing curve, not "
                "a better model."
            ),
            "validation_size": (
                "Thresholds are selected on 163 validation patients, of which "
                "the HER2-positive count is small. A threshold selected on so "
                "few positives is itself high-variance and may not transfer."
            ),
            "no_significance_test": (
                "No hypothesis test or confidence interval was computed for any "
                "threshold difference."
            ),
        },
        "runs": results,
    }

    json_path = OUTPUT_DIR / "threshold_analysis.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = sweep_rows(results)
    csv_path = OUTPUT_DIR / "threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %s and %s (%d sweep rows).", json_path.name, csv_path.name, len(rows))


if __name__ == "__main__":
    main()
