"""Run the active TCGA-BRCA training experiment.

Usage:
    uv run scripts/run_train.py --config configs/breast/train.yaml

The script owns the leakage-critical ordering (CLAUDE.md sections 11-12):

    load cohort -> patient-level split -> fit statistics on the TRAIN fold only
    -> build datasets with those statistics -> train

The fitted statistics are written back into ``config["data"]`` before the
Trainer is constructed, so they are embedded in every checkpoint and inference
reuses them rather than refitting on its own cohort.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.brca_loader import CohortReport, load_brca_cohort  # noqa: E402
from src.data.datasets.multimodal_dataset import (  # noqa: E402
    MultimodalDataset,
    fit_gene_standardization,
)
from src.data.datasets.tabular_dataset import fit_normalization_stats  # noqa: E402
from src.data.pam50 import PAM50_GENES  # noqa: E402
from src.data.schema.patient import Patient  # noqa: E402
from src.data.splits import DataSplit, split_patients, stratum_distribution  # noqa: E402
from src.models.model_factory import build_model  # noqa: E402
from src.training.callbacks import (  # noqa: E402
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)
from src.training.losses import MultiTaskLoss  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.io import ensure_dir, git_commit, load_yaml  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)

#: Filenames written into the run's output directory.
SPLIT_FILENAME = "split.json"
SUMMARY_FILENAME = "run_summary.json"
HISTORY_FILENAME = "history.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/breast/train.yaml",
        help="Path to the training config YAML.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the configured epoch count (for pipeline smoke runs).",
    )
    return parser.parse_args()


def load_merged_config(config_path: str | Path) -> dict[str, Any]:
    """Load train.yaml and merge in the data and model contract files.

    Args:
        config_path: Path to the training config.

    Returns:
        A single config dict with ``"data"`` and ``"model"`` sections. This
        merged dict is what gets embedded in every checkpoint.

    Raises:
        KeyError: If the training config does not name both contract files.
    """
    config = dict(load_yaml(config_path))

    for key in ("data_config", "model_config"):
        if key not in config:
            raise KeyError(
                f"{config_path} must declare '{key}' pointing at its contract file."
            )

    config["data"] = dict(load_yaml(REPO_ROOT / config.pop("data_config")))
    config["model"] = dict(load_yaml(REPO_ROOT / config.pop("model_config")))
    return config


def load_cohort(data_cfg: dict[str, Any]) -> tuple[list[Patient], CohortReport]:
    """Load the cohort from the sources named in the data config."""
    sources = data_cfg.get("sources") or {}
    missing = [
        key
        for key in ("pan_can_clinical", "legacy_clinical", "expression")
        if not sources.get(key)
    ]
    if missing:
        raise KeyError(f"data config 'sources' is missing: {missing}.")

    return load_brca_cohort(
        REPO_ROOT / sources["pan_can_clinical"],
        REPO_ROOT / sources["legacy_clinical"],
        REPO_ROOT / sources["expression"],
    )


def fit_train_fold_statistics(
    split: DataSplit,
    data_cfg: dict[str, Any],
) -> None:
    """Fit preprocessing statistics on the training fold and record them.

    Mutates ``data_cfg`` in place so the statistics travel with the config into
    the checkpoint. Nothing is fitted on the validation or test partitions.
    """
    data_cfg["normalization_stats"] = fit_normalization_stats(
        [patient.clinical for patient in split.train if patient.clinical is not None]
    )
    data_cfg["gene_standardization"] = fit_gene_standardization(split.train)
    data_cfg["gene_order"] = list(PAM50_GENES)

    logger.info(
        "Fitted train-fold statistics on %d patients (age mean=%.2f std=%.2f).",
        len(split.train),
        data_cfg["normalization_stats"]["age"]["mean"],
        data_cfg["normalization_stats"]["age"]["std"],
    )


def build_datasets(
    split: DataSplit,
    data_cfg: dict[str, Any],
) -> dict[str, MultimodalDataset]:
    """Build the three datasets, all using the training fold's statistics."""
    return {
        name: MultimodalDataset(
            patients=patients,
            normalization_stats=data_cfg["normalization_stats"],
            gene_standardization=data_cfg["gene_standardization"],
            gene_order=tuple(data_cfg["gene_order"]),
        )
        for name, patients in split.partitions.items()
    }


def build_loss(config: dict[str, Any]) -> MultiTaskLoss:
    """Build the multi-task loss from the ``loss`` config section.

    Optional per-task class weights may be given as lists; they default to
    ``None``, i.e. focal loss without class reweighting.
    """
    loss_cfg = dict(config.get("loss") or {})

    class_weights = {
        f"{task}_class_weights": _as_weight_tensor(
            loss_cfg.pop(f"{task}_class_weights", None)
        )
        for task in ("subtype", "er", "pr", "her2")
    }

    return MultiTaskLoss(**loss_cfg, **class_weights)


def _as_weight_tensor(values: Any) -> torch.Tensor | None:
    """Convert a config list of class weights to a tensor, or ``None``."""
    if values is None:
        return None
    return torch.tensor([float(v) for v in values], dtype=torch.float32)


def build_callbacks(config: dict[str, Any]) -> list[Any]:
    """Build the checkpoint, early-stopping, and logging callbacks."""
    monitor = str(config.get("monitor", "val_total"))
    mode = str(config.get("monitor_mode", "min"))

    checkpoint_dir = ensure_dir(REPO_ROOT / config.get("checkpoint_dir", "checkpoints"))

    return [
        CheckpointCallback(
            save_dir=checkpoint_dir,
            monitor=monitor,
            mode=mode,
            save_every_n_epochs=config.get("save_every_n_epochs"),
        ),
        EarlyStoppingCallback(
            monitor=monitor,
            patience=int(config.get("early_stopping_patience", 10)),
            mode=mode,
        ),
        LoggingCallback(),
    ]


def write_split(split: DataSplit, output_dir: Path) -> Path:
    """Persist the patient ids of each partition for auditable reuse."""
    path = output_dir / SPLIT_FILENAME
    path.write_text(json.dumps(split.patient_ids(), indent=2), encoding="utf-8")
    logger.info("Split written to %s.", path)
    return path


def write_summary(
    output_dir: Path,
    config: dict[str, Any],
    report: CohortReport,
    split: DataSplit,
    model: torch.nn.Module,
    history: dict[str, list[float]],
) -> Path:
    """Write the reproducibility record for this run (CLAUDE.md section 19)."""
    summary = {
        "experiment_name": config.get("experiment_name"),
        "seed": config.get("seed"),
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
        },
        "config": config,
        "cohort": report.to_dict(),
        "split_sizes": split.sizes(),
        "split_strata": {
            name: stratum_distribution(patients)
            for name, patients in split.partitions.items()
        },
        "model": {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        },
        "epochs_completed": len(next(iter(history.values()), [])),
        "final_metrics": {key: values[-1] for key, values in history.items()},
    }

    path = output_dir / SUMMARY_FILENAME
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (output_dir / HISTORY_FILENAME).write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    logger.info("Run summary written to %s.", path)
    return path


def main() -> None:
    """Load data, train, and record the results."""
    args = parse_args()

    config = load_merged_config(REPO_ROOT / args.config)
    if args.epochs is not None:
        logger.warning("Overriding configured epochs with --epochs %d.", args.epochs)
        config["epochs"] = args.epochs

    set_seed(int(config.get("seed", 42)))

    output_dir = ensure_dir(REPO_ROOT / config.get("output_dir", "results"))
    data_cfg = config["data"]

    patients, report = load_cohort(data_cfg)

    split_cfg = data_cfg.get("split") or {}
    split = split_patients(
        patients,
        val_fraction=float(split_cfg.get("val_fraction", 0.15)),
        test_fraction=float(split_cfg.get("test_fraction", 0.15)),
        seed=int(config.get("seed", 42)),
        stratify=bool(split_cfg.get("stratify", True)),
    )
    write_split(split, output_dir)

    fit_train_fold_statistics(split, data_cfg)
    datasets = build_datasets(split, data_cfg)
    for name, dataset in datasets.items():
        logger.info("%s: %d patients, usable labels %s.", name, len(dataset), dataset.mask_counts())

    model = build_model(config["model"])
    logger.info(
        "Model built: %d parameters.", sum(p.numel() for p in model.parameters())
    )

    trainer = Trainer(
        model=model,
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        loss_fn=build_loss(config),
        config=config,
        callbacks=build_callbacks(config),
    )

    history = trainer.train()
    write_summary(output_dir, config, report, split, model, history)


if __name__ == "__main__":
    main()
