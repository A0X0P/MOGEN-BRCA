"""Compute the full Chapter 4 metric set for the three modality configurations.

Usage:
    uv run scripts/run_detailed_metrics.py

Reads the per-patient test predictions written by
``scripts/export_predictions.py`` and reports, per run and per task, the
quantities the aggregate ``test_metrics.json`` files do not carry: confusion
matrices, balanced accuracy, macro precision/recall, positive-class
precision/recall/specificity/F1, and the survival censoring breakdown.

This script computes nothing from the test labels that could feed back into the
model: it is a reporting pass over predictions that were already fixed by each
run's own best checkpoint. Every scalar that also exists in a committed
``test_metrics.json`` is cross-checked against it, so a disagreement between the
export path and the evaluator fails loudly instead of producing a second,
quietly different set of headline numbers.

Writes ``results/breast_analysis/detailed_metrics.json`` and a flat
``detailed_metrics.csv``. Nothing is written into any training run's directory.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_predictions import SUBTYPE_SLUGS  # noqa: E402
from src.data.pam50 import PAM50_SUBTYPES  # noqa: E402
from src.data.tasks import RECEPTOR_CLASS_LABELS, RECEPTOR_TASKS  # noqa: E402
from src.evaluation import metrics  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

OUTPUT_DIR = REPO_ROOT / "results/breast_analysis"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

#: Largest absolute disagreement tolerated between a metric recomputed here and
#: the same metric in a run's committed ``test_metrics.json``. The two paths run
#: the same metric functions over the same predictions, but the evaluator feeds
#: them float32 tensors while this script reads float64 values back from CSV, so
#: means and sums accumulate at different precision. The largest residual
#: disagreement of that kind is ~9e-9, on Brier score (a mean of squares).
#: A tolerance of 1e-6 admits that while still catching a real divergence: the
#: PR-AUC definition mismatch this check originally caught was 2.2e-1, and the
#: negative-class reconstruction artefact it caught second was 4.4e-5.
CROSS_CHECK_TOLERANCE = 1e-6

#: Why the exported negative-class probability is read from the CSV instead of
#: being reconstructed as ``1 - p_pos``. The two agree to float32 rounding but
#: not in *tie structure*: for the clinical-only HER2 task the model's real
#: ``p(negative)`` column holds 107 distinct values across 134 patients while
#: ``1 - p_pos`` holds 109, because float32 softmax outputs sum to ``1 ± 9e-8``.
#: Average precision collapses tied scores into one threshold, so splitting two
#: tied pairs moved the negative-class AP by 8.8e-5 and the macro by 4.4e-5 —
#: enough to fail the cross-check against the evaluator's own recorded value.
#: Reading the real column reproduces that value exactly.
NEGATIVE_CLASS_NOTE = (
    "The negative-class probability is the model's own softmax output, read "
    "from the exported predictions, not reconstructed as 1 - p_positive. "
    "Reconstruction alters which patients hold identical scores and therefore "
    "perturbs the negative-class average precision at the 1e-4 level."
)

#: What the ``pr_auc`` field of a committed ``test_metrics.json`` actually means
#: for a binary task, established by cross-checking this script's recomputation
#: against those files. The evaluator hands the full ``(N, 2)`` probability
#: matrix to :func:`src.evaluation.metrics.pr_auc`, whose documented behaviour
#: for a 2-D input is to average the per-class average precision. For a binary
#: task that averages the positive-class AP with the negative-class AP, and on
#: an imbalanced task the easy negative class pulls the figure up substantially
#: (HER2: positive-class AP 0.476, negative-class AP 0.917, macro 0.696).
PR_AUC_NOTE = (
    "'pr_auc' in the committed test_metrics.json files is the macro average of "
    "the positive-class and negative-class average precision, because the "
    "evaluator passes the full (N, 2) probability matrix. That is "
    "'pr_auc_macro_both_classes' here. 'pr_auc_positive_class' is the "
    "conventional binary average precision and is the lower, less flattering "
    "figure on the imbalanced HER2 task. Both are reported; neither replaces "
    "the other, and the frozen files were not altered."
)


@dataclass(frozen=True)
class RunSpec:
    """Locates one run's artefacts.

    Attributes:
        name: Identifier used in the output tables.
        label: Human-readable description.
        directory: The run's own output directory.
        frozen: Whether the run is the frozen reference and must not be written to.
    """

    name: str
    label: str
    directory: str
    frozen: bool = False

    @property
    def predictions(self) -> Path:
        return PREDICTIONS_DIR / f"{self.name}_test.csv"

    @property
    def run_summary(self) -> Path:
        return REPO_ROOT / self.directory / "run_summary.json"

    @property
    def test_metrics(self) -> Path:
        return REPO_ROOT / self.directory / "test_metrics.json"

    @property
    def train_log(self) -> Path:
        return REPO_ROOT / self.directory / "train_stdout.log"


RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        name="full_multimodal",
        label="Full multimodal MOGEN-BRCA (gene expression + clinical)",
        directory="results/breast",
        frozen=True,
    ),
    RunSpec(
        name="genomics_only",
        label="Genomics-only ablation (50 PAM50 genes)",
        directory="results/breast_ablation/genomics_only",
    ),
    RunSpec(
        name="clinical_only",
        label="Clinical-only ablation (12-dimensional clinical vector)",
        directory="results/breast_ablation/clinical_only",
    ),
)


def load_predictions(path: Path) -> list[dict[str, str]]:
    """Read a prediction CSV into row dicts."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _masked(rows: list[dict[str, str]], task: str) -> list[dict[str, str]]:
    """Keep only the rows whose mask for ``task`` is set."""
    return [row for row in rows if row[f"{task}_mask"] == "1"]


def subtype_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Full metric set for the 5-class PAM50 task."""
    scored = _masked(rows, "subtype")
    y_true = np.array([int(row["subtype_true"]) for row in scored])
    y_pred = np.array([int(row["subtype_pred"]) for row in scored])
    probs = np.array(
        [[float(row[f"subtype_prob_{slug}"]) for slug in SUBTYPE_SLUGS] for row in scored]
    )

    return {
        "n": int(len(scored)),
        "class_labels": list(PAM50_SUBTYPES),
        "support_per_class": [int((y_true == index).sum()) for index in range(len(PAM50_SUBTYPES))],
        "accuracy": metrics.accuracy(y_true, y_pred),
        "macro_precision": metrics.precision(y_true, y_pred),
        "macro_recall": metrics.recall(y_true, y_pred),
        "balanced_accuracy": metrics.balanced_accuracy(y_true, y_pred),
        "macro_f1": metrics.f1(y_true, y_pred),
        "roc_auc_ovr_macro": metrics.roc_auc(y_true, probs),
        "pr_auc_macro": metrics.pr_auc(y_true, probs),
        "confusion_matrix": metrics.confusion_matrix_counts(
            y_true, y_pred, len(PAM50_SUBTYPES)
        ),
        "confusion_matrix_orientation": "rows = true class, columns = predicted class",
        "majority_class_accuracy": float(
            np.bincount(y_true, minlength=len(PAM50_SUBTYPES)).max() / len(y_true)
        ),
    }


def receptor_metrics(rows: list[dict[str, str]], task: str) -> dict[str, Any]:
    """Full metric set for one binary receptor task.

    Two PR-AUC definitions are reported because they differ materially on the
    imbalanced HER2 task and only one of them is what the committed
    ``test_metrics.json`` files contain. See ``PR_AUC_NOTE``.
    """
    scored = _masked(rows, task)
    y_true = np.array([int(row[f"{task}_true"]) for row in scored])
    y_pred = np.array([int(row[f"{task}_pred"]) for row in scored])
    prob_pos = np.array([float(row[f"{task}_prob_pos"]) for row in scored])
    prob_neg = np.array([float(row[f"{task}_prob_neg"]) for row in scored])
    both_classes = np.column_stack([prob_neg, prob_pos])

    positive = metrics.positive_class_report(y_true, y_pred)

    return {
        "n": int(len(scored)),
        "class_labels": list(RECEPTOR_CLASS_LABELS),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
        "accuracy": metrics.accuracy(y_true, y_pred),
        "macro_precision": metrics.precision(y_true, y_pred),
        "macro_recall": metrics.recall(y_true, y_pred),
        "balanced_accuracy": metrics.balanced_accuracy(y_true, y_pred),
        "macro_f1": metrics.f1(y_true, y_pred),
        "roc_auc": metrics.roc_auc(y_true, prob_pos),
        "pr_auc_positive_class": metrics.pr_auc(y_true, prob_pos),
        "pr_auc_macro_both_classes": metrics.pr_auc(y_true, both_classes),
        "pr_auc_note": PR_AUC_NOTE,
        "negative_class_probability_note": NEGATIVE_CLASS_NOTE,
        "brier_score": metrics.brier_score(y_true, prob_pos),
        "positive_class": positive,
        "confusion_matrix": metrics.confusion_matrix_counts(y_true, y_pred, 2),
        "confusion_matrix_orientation": "rows = true class, columns = predicted class",
        "majority_class_accuracy": float(
            max(int(y_true.sum()), int((y_true == 0).sum())) / len(y_true)
        ),
        "decision_threshold": 0.5,
    }


def survival_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Concordance index with the censoring breakdown."""
    scored = _masked(rows, "survival")
    risk = np.array([float(row["risk_score"]) for row in scored])
    duration = np.array([float(row["duration"]) for row in scored])
    event = np.array([int(row["event"]) for row in scored])

    return {
        "n_evaluable": int(len(scored)),
        "n_events": int(event.sum()),
        "n_censored": int((event == 0).sum()),
        "concordance_index": metrics.concordance_index(risk, duration, event),
        "median_followup_months": float(np.median(duration)),
    }


def cross_check(computed: dict[str, Any], recorded_path: Path) -> dict[str, Any]:
    """Compare recomputed scalars against a run's committed ``test_metrics.json``.

    Args:
        computed: This script's per-task metric mapping.
        recorded_path: Path to the run's committed metrics file.

    Returns:
        A report naming every compared metric and the largest disagreement.
    """
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))

    pairs: list[tuple[str, float, float]] = []
    task_key = {"subtype": "subtype", **{task: task for task in RECEPTOR_TASKS}}

    for task, key in task_key.items():
        block = recorded["classification"].get(key)
        if not block:
            continue
        mine = computed[task]
        pairs.extend(
            [
                (f"{task}.accuracy", mine["accuracy"], block["accuracy"]),
                (f"{task}.macro_precision", mine["macro_precision"], block["precision"]),
                (f"{task}.macro_recall", mine["macro_recall"], block["recall"]),
                (f"{task}.macro_f1", mine["macro_f1"], block["f1"]),
            ]
        )
        auc_key = "roc_auc_ovr_macro" if task == "subtype" else "roc_auc"
        pr_key = "pr_auc_macro" if task == "subtype" else "pr_auc_macro_both_classes"
        if "roc_auc" in block:
            pairs.append((f"{task}.roc_auc", mine[auc_key], block["roc_auc"]))
        if "pr_auc" in block:
            pairs.append((f"{task}.pr_auc", mine[pr_key], block["pr_auc"]))
        if task in RECEPTOR_TASKS:
            pairs.append(
                (
                    f"{task}.brier_score",
                    mine["brier_score"],
                    recorded["calibration"][task]["brier_score"],
                )
            )

    if recorded.get("survival"):
        pairs.append(
            (
                "survival.concordance_index",
                computed["survival"]["concordance_index"],
                recorded["survival"]["concordance_index"],
            )
        )
        pairs.append(
            (
                "survival.n_events",
                float(computed["survival"]["n_events"]),
                recorded["survival"]["n_events"],
            )
        )

    differences = {name: abs(mine - theirs) for name, mine, theirs in pairs}
    worst = max(differences, key=differences.get) if differences else None

    return {
        "recorded_metrics_file": str(recorded_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "metrics_compared": len(pairs),
        "max_abs_difference": float(max(differences.values())) if differences else 0.0,
        "worst_metric": worst,
        "agrees": bool(
            differences and max(differences.values()) <= CROSS_CHECK_TOLERANCE
        ),
        "tolerance": CROSS_CHECK_TOLERANCE,
    }


def log_duration_seconds(path: Path) -> Optional[dict[str, Any]]:
    """Wall-clock span between a training log's first and last timestamp.

    ``run_summary.json`` records no duration field, so the log is the only audit
    trail. The span excludes interpreter start-up and so slightly understates
    total process time; it is reported with that definition attached rather than
    as a clean "training time".

    Args:
        path: Path to a ``train_stdout.log``.

    Returns:
        Mapping with the span and its endpoints, or ``None`` when no log exists.
    """
    if not path.exists():
        return None

    stamps = re.findall(r"^\[(\d{2}):(\d{2}):(\d{2})\]", path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
    if len(stamps) < 2:
        return None

    def as_seconds(stamp: tuple[str, str, str]) -> int:
        hours, minutes, seconds = (int(part) for part in stamp)
        return hours * 3600 + minutes * 60 + seconds

    first, last = as_seconds(stamps[0]), as_seconds(stamps[-1])
    if last < first:  # crossed midnight
        last += 24 * 3600

    return {
        "seconds": last - first,
        "first_log_timestamp": ":".join(stamps[0]),
        "last_log_timestamp": ":".join(stamps[-1]),
        "definition": (
            "Span between the first and last timestamped line of "
            "train_stdout.log. Excludes interpreter start-up."
        ),
    }


def run_provenance(spec: RunSpec) -> dict[str, Any]:
    """Configuration, split, parameter count and runtime for one run."""
    summary = json.loads(spec.run_summary.read_text(encoding="utf-8"))
    config = summary.get("config", {})

    return {
        "experiment_name": summary.get("experiment_name"),
        "output_dir": spec.directory,
        "frozen_reference": spec.frozen,
        "git_commit": summary.get("git_commit"),
        "seed": summary.get("seed"),
        "split_sizes": summary.get("split_sizes"),
        "trainable_parameters": summary.get("model", {}).get("trainable_parameters"),
        "total_parameters": summary.get("model", {}).get("total_parameters"),
        "epochs_completed": summary.get("epochs_completed"),
        "training_protocol": {
            key: config.get(key)
            for key in (
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "scheduler",
                "monitor",
                "monitor_mode",
                "early_stopping_patience",
                "mixed_precision",
                "device",
            )
        },
        "training_duration": log_duration_seconds(spec.train_log),
        "environment": summary.get("environment"),
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


def evaluate_run(spec: RunSpec) -> dict[str, Any]:
    """Compute every reported metric for one run."""
    rows = load_predictions(spec.predictions)

    tasks: dict[str, Any] = {"subtype": subtype_metrics(rows)}
    for task in RECEPTOR_TASKS:
        tasks[task] = receptor_metrics(rows, task)
    tasks["survival"] = survival_metrics(rows)

    return {
        "label": spec.label,
        "provenance": run_provenance(spec),
        "cross_check_against_recorded_metrics": cross_check(tasks, spec.test_metrics),
        "n_test_patients": len(rows),
        "tasks": tasks,
    }


def flat_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the nested report into one CSV row per run and task."""
    rows: list[dict[str, Any]] = []

    for run_name, run in results.items():
        for task, block in run["tasks"].items():
            row: dict[str, Any] = {"run": run_name, "task": task}
            for key, value in block.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[key] = value
            if task in RECEPTOR_TASKS:
                for key, value in block["positive_class"].items():
                    row[f"positive_{key}"] = value
            rows.append(row)

    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    return [{column: row.get(column, "") for column in columns} for row in rows]


def main() -> None:
    """Compute and write the detailed metric report."""
    ensure_dir(OUTPUT_DIR)

    results = {spec.name: evaluate_run(spec) for spec in RUNS}

    for name, run in results.items():
        check = run["cross_check_against_recorded_metrics"]
        logger.info(
            "%s: %d metrics cross-checked against %s, max abs diff %.3e, agrees=%s.",
            name,
            check["metrics_compared"],
            check["recorded_metrics_file"],
            check["max_abs_difference"],
            check["agrees"],
        )
        if not check["agrees"]:
            raise ValueError(
                f"Recomputed metrics for '{name}' disagree with "
                f"{check['recorded_metrics_file']} (worst: {check['worst_metric']}, "
                f"{check['max_abs_difference']:.3e} > {check['tolerance']:.0e}). "
                "The prediction export and the evaluator have diverged; fix that "
                "before reporting either set of numbers."
            )

    payload = {
        "purpose": (
            "Per-task metric set for the three modality configurations, "
            "including the confusion matrices, balanced accuracy and "
            "positive-class breakdowns that the aggregate test_metrics.json "
            "files do not carry."
        ),
        "partition": "test",
        "analysis_git_commit": git_commit(),
        "predictions_source": "scripts/export_predictions.py",
        "notes": {
            "balanced_accuracy": (
                "Equals macro-averaged recall for a single-label task; both are "
                "reported so neither convention has to be inferred."
            ),
            "positive_class": (
                "Reported for all three receptor tasks. Required for HER2, "
                "which is the imbalanced task, and given for ER/PR so the three "
                "are directly comparable."
            ),
            "threshold": (
                "All metrics here use the default 0.5 decision threshold. The "
                "validation-selected alternatives are in threshold_analysis.json."
            ),
            "pr_auc": PR_AUC_NOTE,
            "negative_class_probability": NEGATIVE_CLASS_NOTE,
            "statistical_testing": (
                "No hypothesis test was performed. Differences between runs must "
                "not be described as statistically significant."
            ),
        },
        "runs": results,
    }

    json_path = OUTPUT_DIR / "detailed_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = flat_rows(results)
    csv_path = OUTPUT_DIR / "detailed_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %s and %s (%d rows).", json_path.name, csv_path.name, len(rows))


if __name__ == "__main__":
    main()
