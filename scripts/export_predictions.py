"""Export per-patient predictions for one checkpoint and one partition.

Usage:
    uv run scripts/export_predictions.py \
        --config configs/breast/train.yaml \
        --checkpoint results/breast/checkpoints/checkpoint_best.pt \
        --partition val \
        --output results/breast_analysis/predictions/full_multimodal_val.csv

Why this exists
---------------
``scripts/run_eval.py`` writes aggregate metrics into the *training run's* own
output directory. Two of the Chapter 4 analyses need something it does not
produce, and must not write where it writes:

* confusion matrices and positive-class metrics need the per-patient
  predictions, not the aggregates;
* the HER2/PR threshold analysis needs *validation* predictions, and the frozen
  reference run's directory (``results/breast/``) must stay byte-for-byte
  unchanged.

So this script takes an explicit ``--output`` path and never touches the
training run's directory. It is read-only with respect to the checkpoint: the
model is loaded, set to eval mode, and run under ``torch.no_grad()``.

Preprocessing statistics come from the checkpoint. Nothing is refitted, so
exporting validation or test predictions cannot leak either partition into the
train-fold statistics.

Masked-out labels are written as empty cells, never as the ``IGNORE_INDEX``
sentinel, so a downstream reader cannot mistake an absent label for a class.

Both class probabilities are written for the binary receptor tasks, even though
one looks recoverable from the other. It is not, exactly: the model's softmax is
computed in float32, where the two columns sum to ``1 ± 9e-8``, and
reconstructing ``p(negative)`` as ``1 - p(positive)`` in float64 changes which
patients hold *identical* scores. Average precision collapses tied scores into a
single threshold, so a reconstructed column shifts the negative-class
average precision by ~1e-4 and makes the exported predictions fail to reproduce
the evaluator's own recorded metrics. Storing the real column removes that.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_eval import (  # noqa: E402
    build_dataloader,
    rebuild_split,
    verify_against_training_split,
)
from scripts.run_train import load_merged_config  # noqa: E402
from src.data.pam50 import PAM50_SUBTYPES  # noqa: E402
from src.data.tasks import (  # noqa: E402
    CLASSIFICATION_TASKS,
    RECEPTOR_TASKS,
    RISK_SCORE_KEY,
    SURVIVAL_TASK,
    TASK_LOGIT_KEYS,
)
from src.evaluation.evaluator import build_model_inputs  # noqa: E402
from src.inference.predict import InferenceArtifacts, load_model  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)

#: Column-name slug per PAM50 class, used for the subtype probability columns.
SUBTYPE_SLUGS: tuple[str, ...] = tuple(
    name.lower().replace("-", "_").replace(" ", "_") for name in PAM50_SUBTYPES
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Training config the checkpoint was produced with.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to run.")
    parser.add_argument(
        "--partition",
        required=True,
        choices=("train", "val", "test"),
        help="Partition to export predictions for.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination CSV path. Chosen explicitly so that nothing is "
        "written into a frozen run directory.",
    )
    return parser.parse_args()


def field_names() -> list[str]:
    """Return the CSV column order."""
    columns = ["patient_id"]

    columns.append("subtype_mask")
    columns.extend(["subtype_true", "subtype_pred"])
    columns.extend(f"subtype_prob_{slug}" for slug in SUBTYPE_SLUGS)

    for task in RECEPTOR_TASKS:
        columns.extend(
            [
                f"{task}_mask",
                f"{task}_true",
                f"{task}_pred",
                f"{task}_prob_neg",
                f"{task}_prob_pos",
            ]
        )

    columns.extend(["survival_mask", "risk_score", "duration", "event"])
    return columns


def predict_batches(
    artifacts: InferenceArtifacts, dataloader: Any
) -> Iterator[dict[str, Any]]:
    """Yield one row dict per patient, in dataloader order.

    Args:
        artifacts: Loaded checkpoint artifacts.
        dataloader: Unshuffled loader over a single partition.

    Yields:
        One dict per patient, keyed by :func:`field_names`.
    """
    model = artifacts.model
    model.eval()
    modalities = getattr(model, "active_modalities", None)

    with torch.no_grad():
        for batch in dataloader:
            output = model(**build_model_inputs(batch, artifacts.device, modalities))
            yield from _rows_for_batch(batch, output)


def _rows_for_batch(
    batch: dict[str, Any], output: dict[str, torch.Tensor]
) -> Iterator[dict[str, Any]]:
    """Convert one model output batch into per-patient row dicts."""
    probs = {
        task: torch.softmax(output[TASK_LOGIT_KEYS[task]].float().cpu(), dim=-1)
        for task in CLASSIFICATION_TASKS
        if TASK_LOGIT_KEYS[task] in output
    }
    risk = (
        output[RISK_SCORE_KEY].reshape(-1).float().cpu()
        if RISK_SCORE_KEY in output
        else None
    )

    for position, patient_id in enumerate(batch["patient_id"]):
        row: dict[str, Any] = {"patient_id": patient_id}

        for task in CLASSIFICATION_TASKS:
            masked_in = bool(batch["mask"][task][position])
            row[f"{task}_mask"] = int(masked_in)

            task_probs = probs.get(task)
            if task_probs is None:
                continue

            row[f"{task}_pred"] = int(task_probs[position].argmax())
            row[f"{task}_true"] = (
                int(batch["label"][task][position]) if masked_in else ""
            )

            if task == "subtype":
                for index, slug in enumerate(SUBTYPE_SLUGS):
                    row[f"subtype_prob_{slug}"] = float(task_probs[position, index])
            else:
                # Both columns as the model produced them; see the module
                # docstring for why the negative column is not reconstructed.
                row[f"{task}_prob_neg"] = float(task_probs[position, 0])
                row[f"{task}_prob_pos"] = float(task_probs[position, 1])

        survival_in = bool(batch["mask"][SURVIVAL_TASK][position])
        row["survival_mask"] = int(survival_in)
        row["risk_score"] = float(risk[position]) if risk is not None else ""
        row["duration"] = (
            float(batch[SURVIVAL_TASK]["duration"][position]) if survival_in else ""
        )
        row["event"] = (
            int(batch[SURVIVAL_TASK]["event"][position]) if survival_in else ""
        )

        yield row


def checkpoint_epoch(path: Path) -> int:
    """Read the epoch a checkpoint was saved at (0-based, as stored)."""
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    artifacts: InferenceArtifacts,
    epoch: int,
    n_rows: int,
) -> Path:
    """Record what produced a prediction file, next to the file itself."""
    manifest = {
        "predictions_file": path.name,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": epoch,
        "partition": args.partition,
        "n_patients": n_rows,
        "gene_order": list(artifacts.gene_order),
        "preprocessing": (
            "Train-fold statistics read from the checkpoint; nothing refitted."
        ),
        "model_trainable_parameters": sum(
            parameter.numel()
            for parameter in artifacts.model.parameters()
            if parameter.requires_grad
        ),
        "active_modalities": list(
            getattr(artifacts.model, "active_modalities", ()) or ()
        ),
    }
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    """Export per-patient predictions for one checkpoint and partition."""
    args = parse_args()

    config = load_merged_config(REPO_ROOT / args.config)
    set_seed(int(config.get("seed", 42)))

    artifacts = load_model(REPO_ROOT / args.checkpoint)

    split = rebuild_split(config)
    verify_against_training_split(
        split, REPO_ROOT / config.get("output_dir", "results")
    )

    loader = build_dataloader(
        split.partitions[args.partition],
        artifacts,
        batch_size=int(config.get("batch_size", 32)),
    )

    rows = list(predict_batches(artifacts, loader))

    output_path = REPO_ROOT / args.output
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names())
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = write_manifest(
        output_path, args, artifacts, checkpoint_epoch(REPO_ROOT / args.checkpoint), len(rows)
    )
    logger.info(
        "Wrote %d %s predictions to %s (manifest: %s).",
        len(rows),
        args.partition,
        output_path,
        manifest_path.name,
    )


if __name__ == "__main__":
    main()
