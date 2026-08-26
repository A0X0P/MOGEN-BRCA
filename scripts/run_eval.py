"""Evaluate a trained TCGA-BRCA checkpoint on the held-out test partition.

Usage:
    uv run scripts/run_eval.py --config configs/breast/train.yaml \
        --checkpoint results/breast/checkpoints/checkpoint_best.pt

The split is recomputed deterministically from the same cohort, seed, and
fractions used at training time. When the training run's ``split.json`` is
present it is cross-checked, so a configuration drift that would silently
evaluate on training patients fails loudly instead.

Preprocessing statistics are read from the checkpoint, never refitted here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_train import (  # noqa: E402
    SPLIT_FILENAME,
    load_cohort,
    load_merged_config,
)
from src.data.datasets.multimodal_dataset import MultimodalDataset  # noqa: E402
from src.data.pam50 import PAM50_GENES  # noqa: E402
from src.data.schema.patient import Patient  # noqa: E402
from src.data.splits import DataSplit, split_patients  # noqa: E402
from src.evaluation.evaluator import EvaluationResult, evaluate  # noqa: E402
from src.inference.predict import InferenceArtifacts, load_model  # noqa: E402
from src.training.trainer import collate_multimodal  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)

#: Suffix of the metrics report written into the output directory. The
#: evaluated partition is prefixed, giving ``test_metrics.json`` /
#: ``val_metrics.json`` / ``train_metrics.json``.
METRICS_FILENAME = "metrics.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/breast/train.yaml",
        help="Path to the training config the checkpoint was produced with.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint to evaluate.",
    )
    parser.add_argument(
        "--partition",
        default="test",
        choices=("train", "val", "test"),
        help="Which partition to evaluate. Defaults to the held-out test set.",
    )
    return parser.parse_args()


def rebuild_split(config: dict[str, Any]) -> DataSplit:
    """Recompute the deterministic patient-level split for this config."""
    patients, report = load_cohort(config["data"])
    report.log()

    split_cfg = config["data"].get("split") or {}
    return split_patients(
        patients,
        val_fraction=float(split_cfg.get("val_fraction", 0.15)),
        test_fraction=float(split_cfg.get("test_fraction", 0.15)),
        seed=int(config.get("seed", 42)),
        stratify=bool(split_cfg.get("stratify", True)),
    )


def verify_against_training_split(split: DataSplit, output_dir: Path) -> None:
    """Check the recomputed split matches the one the run trained on.

    Args:
        split: Freshly recomputed split.
        output_dir: Training run's output directory.

    Raises:
        ValueError: If any partition's patient set differs from the recorded
            split, which would mean the checkpoint saw these patients.
    """
    path = output_dir / SPLIT_FILENAME
    if not path.exists():
        logger.warning(
            "No %s found; evaluating against the recomputed split without "
            "cross-checking it against the training run.",
            path,
        )
        return

    recorded: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
    current = split.patient_ids()

    for name, ids in recorded.items():
        if set(ids) != set(current.get(name, [])):
            raise ValueError(
                f"Recomputed '{name}' partition ({len(current.get(name, []))} "
                f"patients) does not match {path} ({len(ids)} patients). The "
                "cohort, seed, or split fractions have changed since training; "
                "evaluating would mix training patients into the test set."
            )

    logger.info("Recomputed split matches the training run's %s.", path)


def build_dataloader(
    patients: list[Patient],
    artifacts: InferenceArtifacts,
    batch_size: int,
) -> DataLoader:
    """Build a loader over one partition using the checkpoint's statistics."""
    dataset = MultimodalDataset(
        patients=patients,
        normalization_stats=artifacts.normalization_stats,
        gene_standardization=artifacts.gene_standardization,
        gene_order=artifacts.gene_order or PAM50_GENES,
    )
    logger.info(
        "Evaluating %d patients; usable labels per task: %s.",
        len(dataset),
        dataset.mask_counts(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multimodal,
    )


def write_metrics(
    result: EvaluationResult,
    output_dir: Path,
    partition: str,
    checkpoint_path: str,
) -> Path:
    """Write the evaluation report to disk and log its headline metrics."""
    payload = {
        "partition": partition,
        "checkpoint": checkpoint_path,
        **result.to_dict(),
    }

    path = output_dir / f"{partition}_{METRICS_FILENAME}"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    for task, metrics in result.classification.items():
        logger.info(
            "%s: n=%d %s",
            task,
            result.task_counts.get(task, 0),
            " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())),
        )
    if result.survival:
        logger.info(
            "survival: %s",
            " ".join(f"{k}={v:.4f}" for k, v in sorted(result.survival.items())),
        )

    logger.info("Metrics written to %s.", path)
    return path


def main() -> None:
    """Load a checkpoint and evaluate it on the requested partition."""
    args = parse_args()

    config = load_merged_config(REPO_ROOT / args.config)
    set_seed(int(config.get("seed", 42)))

    output_dir = ensure_dir(REPO_ROOT / config.get("output_dir", "results"))

    artifacts = load_model(REPO_ROOT / args.checkpoint)
    if artifacts.gene_standardization is None:
        logger.warning(
            "Checkpoint records no gene standardization; evaluating on "
            "unstandardised expression values, which will not match training."
        )

    split = rebuild_split(config)
    verify_against_training_split(split, output_dir)

    loader = build_dataloader(
        split.partitions[args.partition],
        artifacts,
        batch_size=int(config.get("batch_size", 32)),
    )

    result = evaluate(artifacts.model, loader, device=artifacts.device)

    write_metrics(result, output_dir, args.partition, str(args.checkpoint))


if __name__ == "__main__":
    main()
