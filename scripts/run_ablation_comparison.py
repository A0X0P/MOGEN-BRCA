"""Build the machine-readable full-vs-ablation comparison (Chapter 4, Objective 3).

Usage:
    uv run scripts/run_ablation_comparison.py

Reads the three already-written ``test_metrics.json`` reports and emits:

    results/breast_ablation/ablation_comparison.json
    results/breast_ablation/ablation_comparison.csv

This script is read-only with respect to every run it compares: it never
retrains, never re-evaluates, and never writes inside ``results/breast/``.

Split integrity
---------------
Every comparison is meaningless unless the three runs share one held-out test
partition. Each run writes its own ``split.json``; this script asserts all of
them are order-identical to the frozen run's before emitting anything, and
records that verification in the output.

Metric provenance
-----------------
No new metric is implemented here. ``balanced_accuracy`` is the evaluator's
existing macro-averaged ``recall``: for a single-label task, macro recall is
by definition the mean per-class recall, which is balanced accuracy. The
majority-class accuracy of the test partition is included as a reference
column, because accuracy alone cannot distinguish a real classifier from one
that has collapsed onto the majority class.

No statistical test is performed and no significance is claimed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_eval import rebuild_split  # noqa: E402
from scripts.run_train import load_merged_config  # noqa: E402
from src.data.tasks import CLASSIFICATION_TASKS, SURVIVAL_TASK  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

#: Output directory for the ablation comparison. Never results/breast/.
OUTPUT_DIR = REPO_ROOT / "results/breast_ablation"

#: The frozen full model is the reference arm; the deltas are Full - ablation.
REFERENCE_MODEL = "full_multimodal"

#: Each arm: the run directory and the training config that produced it.
MODELS: dict[str, dict[str, str]] = {
    "full_multimodal": {
        "run_dir": "results/breast",
        "config": "configs/breast/train.yaml",
        "checkpoint": "results/breast/checkpoints/checkpoint_best.pt",
        "description": "Genomic Transformer + Clinical MLP + cross-modal attention + fusion",
    },
    "genomics_only": {
        "run_dir": "results/breast_ablation/genomics_only",
        "config": "configs/breast/train_genomics_only.yaml",
        "checkpoint": "results/breast_ablation/genomics_only/checkpoints/checkpoint_best.pt",
        "description": "Genomic Transformer + fusion; no clinical branch, no cross-modal attention",
    },
    "clinical_only": {
        "run_dir": "results/breast_ablation/clinical_only",
        "config": "configs/breast/train_clinical_only.yaml",
        "checkpoint": "results/breast_ablation/clinical_only/checkpoints/checkpoint_best.pt",
        "description": "Clinical MLP + fusion; no genomic branch, no cross-modal attention",
    },
}

#: Comparison columns, in report order. ``brier_score`` applies to the binary
#: receptor tasks only; ``concordance_index`` to survival only.
CLASSIFICATION_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
)
CSV_COLUMNS = (
    "model",
    "task",
    "n",
    *CLASSIFICATION_COLUMNS,
    "concordance_index",
    "n_events",
)

#: Maps the comparison column onto the key the evaluator actually wrote.
_SOURCE_KEY = {
    "accuracy": "accuracy",
    "balanced_accuracy": "recall",
    "macro_f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "pr_auc",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        default="test",
        help="Which partition's metrics report to compare. Defaults to test.",
    )
    return parser.parse_args()


def load_run(name: str, spec: dict[str, str], partition: str) -> dict[str, Any]:
    """Load one arm's metrics, summary, history, and best epoch.

    Args:
        name: Arm name.
        spec: Entry from :data:`MODELS`.
        partition: Partition whose ``<partition>_metrics.json`` to read.

    Returns:
        The arm's raw artifacts and derived training facts.

    Raises:
        FileNotFoundError: If the run has not been evaluated on ``partition``.
    """
    run_dir = REPO_ROOT / spec["run_dir"]
    metrics_path = run_dir / f"{partition}_metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{name} has no {metrics_path.name}. Evaluate it first with "
            f"run_eval.py --config {spec['config']} --checkpoint {spec['checkpoint']}."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))

    checkpoint = torch.load(
        REPO_ROOT / spec["checkpoint"], map_location="cpu", weights_only=False
    )

    val_total = history["val_total"]
    best_epoch_from_history = min(range(len(val_total)), key=val_total.__getitem__)

    return {
        "metrics": metrics,
        "summary": summary,
        "best_epoch": int(checkpoint["epoch"]),
        "best_epoch_from_history": best_epoch_from_history,
        "best_val_total": float(val_total[best_epoch_from_history]),
        "epochs_completed": len(val_total),
        "epochs_configured": int(summary["config"]["epochs"]),
        "early_stopping_patience": int(summary["config"]["early_stopping_patience"]),
        "monitor": str(summary["config"]["monitor"]),
        "total_parameters": int(summary["model"]["total_parameters"]),
        "trainable_parameters": int(summary["model"]["trainable_parameters"]),
        "git_commit": summary.get("git_commit"),
        "split": json.loads((run_dir / "split.json").read_text(encoding="utf-8")),
    }


def verify_shared_split(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Assert every arm trained and was evaluated on one identical split.

    Args:
        runs: Loaded arms, keyed by name.

    Returns:
        A record of the verification for the output file.

    Raises:
        ValueError: If any arm's split differs from the reference arm's, in
            either membership or order.
    """
    reference = runs[REFERENCE_MODEL]["split"]

    for name, run in runs.items():
        if name == REFERENCE_MODEL:
            continue
        for partition, ids in reference.items():
            other = run["split"].get(partition, [])
            if other != ids:
                raise ValueError(
                    f"{name} '{partition}' split differs from {REFERENCE_MODEL}: "
                    f"{len(other)} vs {len(ids)} patients (or a different order). "
                    "The ablation comparison would not be like-for-like."
                )

    overlaps = {
        f"{a}_and_{b}": len(set(reference[a]) & set(reference[b]))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlaps.values()):
        raise ValueError(f"Patient overlap between partitions: {overlaps}.")

    logger.info(
        "Split verified identical across %d arms: %s.",
        len(runs),
        {k: len(v) for k, v in reference.items()},
    )

    return {
        "all_arms_share_one_split": True,
        "order_identical": True,
        "partition_sizes": {k: len(v) for k, v in reference.items()},
        "pairwise_patient_overlap": overlaps,
        "reference": f"{MODELS[REFERENCE_MODEL]['run_dir']}/split.json",
    }


def majority_class_accuracy(partition: str) -> dict[str, Optional[float]]:
    """Accuracy a constant majority-class predictor would score per task.

    This is a descriptive property of the evaluated partition's labels, used
    only to make the accuracy column interpretable. It tunes nothing.

    Args:
        partition: Partition to describe.

    Returns:
        Mapping of task name to the majority-class rate, or ``None`` for
        survival (where accuracy is not defined).
    """
    config = load_merged_config(REPO_ROOT / MODELS[REFERENCE_MODEL]["config"])
    split = rebuild_split(config)
    patients = split.partitions[partition]

    getters = {
        "subtype": lambda p: p.targets.subtype_index,
        "er": lambda p: p.targets.er_positive,
        "pr": lambda p: p.targets.pr_positive,
        "her2": lambda p: p.targets.her2_positive,
    }

    rates: dict[str, Optional[float]] = {}
    for task, get in getters.items():
        labels = [get(p) for p in patients]
        present = [value for value in labels if value is not None]
        counts: dict[Any, int] = {}
        for value in present:
            counts[value] = counts.get(value, 0) + 1
        rates[task] = max(counts.values()) / len(present) if present else None

    rates[SURVIVAL_TASK] = None
    return rates


def task_row(name: str, run: dict[str, Any], task: str) -> dict[str, Any]:
    """Extract one arm's metrics for one task into comparison columns."""
    metrics = run["metrics"]
    row: dict[str, Any] = {"model": name, "task": task}
    row["n"] = metrics["task_counts"].get(task)

    if task == SURVIVAL_TASK:
        survival = metrics.get("survival") or {}
        for column in CLASSIFICATION_COLUMNS:
            row[column] = None
        row["concordance_index"] = survival.get("concordance_index")
        row["n_events"] = survival.get("n_events")
        return row

    task_metrics = metrics["classification"].get(task, {})
    for column in CLASSIFICATION_COLUMNS:
        if column == "brier_score":
            row[column] = (metrics.get("calibration", {}).get(task, {})).get(
                "brier_score"
            )
        else:
            row[column] = task_metrics.get(_SOURCE_KEY[column])

    row["concordance_index"] = None
    row["n_events"] = None
    return row


def difference_rows(rows: list[dict[str, Any]], ablation: str) -> list[dict[str, Any]]:
    """Compute ``Full - ablation`` absolute differences, task by task."""
    by_key = {(row["model"], row["task"]): row for row in rows}
    numeric = (*CLASSIFICATION_COLUMNS, "concordance_index")

    out: list[dict[str, Any]] = []
    for task in (*CLASSIFICATION_TASKS, SURVIVAL_TASK):
        reference = by_key[(REFERENCE_MODEL, task)]
        other = by_key[(ablation, task)]

        if reference["n"] != other["n"]:
            raise ValueError(
                f"Task '{task}' was scored on {reference['n']} patients for "
                f"{REFERENCE_MODEL} but {other['n']} for {ablation}; the "
                "difference would not be like-for-like."
            )

        row: dict[str, Any] = {
            "model": f"{REFERENCE_MODEL}_minus_{ablation}",
            "task": task,
            "n": reference["n"],
            "n_events": reference["n_events"],
        }
        for column in numeric:
            a, b = reference[column], other[column]
            row[column] = None if a is None or b is None else a - b
        out.append(row)

    return out


def build_payload(
    runs: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    split_check: dict[str, Any],
    majority: dict[str, Optional[float]],
    partition: str,
) -> dict[str, Any]:
    """Assemble the JSON comparison document."""
    by_model: dict[str, Any] = {}
    for name, run in runs.items():
        by_model[name] = {
            "description": MODELS[name]["description"],
            "run_dir": MODELS[name]["run_dir"],
            "config": MODELS[name]["config"],
            "checkpoint": MODELS[name]["checkpoint"],
            "training": {
                "best_epoch": run["best_epoch"],
                "best_epoch_from_history_argmin": run["best_epoch_from_history"],
                "best_val_total": run["best_val_total"],
                "epochs_completed": run["epochs_completed"],
                "epochs_configured": run["epochs_configured"],
                "early_stopped": run["epochs_completed"] < run["epochs_configured"],
                "early_stopping_patience": run["early_stopping_patience"],
                "model_selection_monitor": run["monitor"],
                "seed": run["summary"]["config"]["seed"],
                "git_commit": run["git_commit"],
            },
            "parameters": {
                "total": run["total_parameters"],
                "trainable": run["trainable_parameters"],
            },
            "tasks": {
                row["task"]: {k: v for k, v in row.items() if k not in ("model", "task")}
                for row in rows
                if row["model"] == name
            },
        }

    differences = {
        row["model"]: {}
        for row in rows
        if row["model"].startswith(f"{REFERENCE_MODEL}_minus_")
    }
    for row in rows:
        if row["model"] in differences:
            differences[row["model"]][row["task"]] = {
                k: v for k, v in row.items() if k not in ("model", "task")
            }

    return {
        "purpose": (
            "Determine whether two-modality fusion adds predictive value over "
            "each single-modality ablation on one shared held-out partition."
        ),
        "partition": partition,
        "reference_model": REFERENCE_MODEL,
        "difference_convention": (
            f"{REFERENCE_MODEL} minus ablation. Positive favours the full "
            "multimodal model. For brier_score, LOWER is better, so a positive "
            "difference favours the ablation."
        ),
        "split_verification": split_check,
        "metric_notes": {
            "balanced_accuracy": (
                "The evaluator's macro-averaged 'recall'. For a single-label "
                "task macro recall is the mean per-class recall, i.e. balanced "
                "accuracy. No new metric implementation was introduced."
            ),
            "macro_f1": "Macro-averaged F1 over all classes of the task.",
            "pr_auc": (
                "Macro-averaged average-precision over classes (both classes "
                "for the binary receptor tasks, all five for PAM50). This is "
                "not the positive-class average precision."
            ),
            "roc_auc": "One-vs-rest ROC-AUC, macro-averaged for PAM50.",
            "brier_score": (
                "Positive-class Brier score; receptor tasks only. Lower is better."
            ),
            "concordance_index": (
                "Harrell's C-index on the survival-evaluable subset of the "
                "partition. Survival is the only task with an n_events figure."
            ),
            "majority_class_accuracy": (
                "Accuracy of a constant majority-class predictor on this "
                "partition. A descriptive property of the labels, included so "
                "the accuracy column cannot be misread: an arm whose accuracy "
                "equals this value and whose balanced_accuracy is 0.5 has "
                "collapsed onto the majority class and learned nothing."
            ),
        },
        "test_partition_majority_class_accuracy": majority,
        "models": by_model,
        "differences": differences,
        "statistical_testing": {
            "performed": False,
            "tests": [],
            "note": (
                "No hypothesis test, confidence interval, or resampling "
                "procedure was run. All differences are point estimates from a "
                "single seed, a single split, and one training run per arm. No "
                "statistical significance is claimed."
            ),
        },
        "caveats": [
            "PAM50 is reproduction/recovery of the published PAM50 assignment "
            "from the same 50-gene panel the assignment derives from. It is not "
            "independent molecular-subtype discovery or independent phenotype "
            "prediction.",
            "Each arm selected its own checkpoint by minimum val_total under an "
            "identical protocol, so the arms stopped at different epochs. The "
            "measured differences therefore reflect modality AND the "
            "selection epoch that modality's validation curve produced, not "
            "modality in isolation.",
            "val_total is dominated by the Cox term, so model selection is "
            "weighted towards survival for every arm.",
            "No external cohort was evaluated. No claim of external "
            "generalisation or clinical utility is supported.",
        ],
    }


def write_outputs(payload: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Write the JSON and CSV comparison files."""
    ensure_dir(OUTPUT_DIR)

    json_path = OUTPUT_DIR / "ablation_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = OUTPUT_DIR / "ablation_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})

    logger.info("Comparison written to %s and %s.", json_path, csv_path)
    return json_path, csv_path


def main() -> None:
    """Compare the frozen full model against both single-modality ablations."""
    args = parse_args()

    runs = {
        name: load_run(name, spec, args.partition) for name, spec in MODELS.items()
    }
    split_check = verify_shared_split(runs)
    majority = majority_class_accuracy(args.partition)

    rows: list[dict[str, Any]] = []
    for name, run in runs.items():
        for task in (*CLASSIFICATION_TASKS, SURVIVAL_TASK):
            rows.append(task_row(name, run, task))

    for ablation in ("genomics_only", "clinical_only"):
        rows.extend(difference_rows(rows, ablation))

    payload = build_payload(runs, rows, split_check, majority, args.partition)
    write_outputs(payload, rows)

    for name in MODELS:
        training = payload["models"][name]["training"]
        logger.info(
            "%s: best_epoch=%d best_val_total=%.6f params=%d",
            name,
            training["best_epoch"],
            training["best_val_total"],
            payload["models"][name]["parameters"]["total"],
        )


if __name__ == "__main__":
    main()
