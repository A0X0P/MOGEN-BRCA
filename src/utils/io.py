"""File I/O helpers: checkpoint save/load, YAML config loading, path utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents as a dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid YAML or does not parse to a dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        contents = yaml.safe_load(f)

    if not isinstance(contents, dict):
        raise ValueError(
            f"Expected YAML dict at {path}, got {type(contents).__name__}."
        )
    return contents


def _get_git_commit() -> Optional[str]:
    """Return the current HEAD commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def git_commit() -> Optional[str]:
    """Return the current HEAD commit hash for run provenance.

    Returns:
        The full commit hash, or ``None`` outside a git checkout or when git is
        unavailable. Recorded in run summaries and checkpoints (CLAUDE.md
        section 19).
    """
    return _get_git_commit()


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Save a training checkpoint to disk.

    Injects the current git commit hash under ``"git_commit"`` if absent.

    Args:
        state: Dict containing model state, optimizer state, epoch, config,
            and any other fields to persist.
        path: Destination file path (parent directories are created if needed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "git_commit" not in state:
        state = {**state, "git_commit": _get_git_commit()}

    torch.save(state, path)
    logger.info("Checkpoint saved to %s.", path)


def load_checkpoint(
    path: str | Path,
    map_location: Optional[Any] = None,
) -> dict[str, Any]:
    """Load a checkpoint from disk.

    Args:
        path: Path to the checkpoint file.
        map_location: Passed to ``torch.load`` for device remapping.
            Pass ``"cpu"`` to load a GPU checkpoint on a CPU-only machine.

    Returns:
        The checkpoint dict as saved by :func:`save_checkpoint`.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the loaded object is not a dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected checkpoint dict at {path}, got {type(checkpoint).__name__}."
        )

    commit = checkpoint.get("git_commit") or "?"
    logger.info(
        "Loaded checkpoint from %s (epoch=%s, git=%s).",
        path,
        checkpoint.get("epoch", "?"),
        commit[:8] if commit != "?" else "?",
    )
    return checkpoint


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
