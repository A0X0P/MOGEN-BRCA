"""Reproducibility helpers: deterministic seeding for all RNG sources.

Calling set_seed once at the start of a training run ensures identical results
across runs with the same seed. Setting cudnn.deterministic=True may slow
training on GPU — this is intentional; reproducibility is prioritised over
throughput per the project's design philosophy.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Seed all RNG sources for deterministic execution.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU and CUDA). Enables
    cuDNN deterministic mode and disables benchmark mode so convolution
    algorithms are fixed across runs.

    Args:
        seed: Integer seed value. Use the same value to reproduce a run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d.", seed)
