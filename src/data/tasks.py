"""Task vocabulary and masking constants for the active BRCA pipeline.

Single source of truth for the five prediction tasks, their class counts, and
the sentinel used for absent classification labels. The dataset, loss,
trainer, and evaluator all import from here so that a task can never be named
inconsistently across layers.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from src.data.pam50 import PAM50_SUBTYPES

#: Sentinel written into a classification target when no usable label exists.
#: Matches ``torch.nn.functional.cross_entropy``'s default ``ignore_index`` so
#: that a mask bug degrades into a loud error rather than a silent wrong label.
IGNORE_INDEX: Final[int] = -100

#: The four classification tasks, in canonical order.
CLASSIFICATION_TASKS: Final[tuple[str, ...]] = ("subtype", "er", "pr", "her2")

#: The survival task name.
SURVIVAL_TASK: Final[str] = "survival"

#: All five tasks, in canonical order.
ALL_TASKS: Final[tuple[str, ...]] = (*CLASSIFICATION_TASKS, SURVIVAL_TASK)

#: Number of classes per classification task.
TASK_NUM_CLASSES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "subtype": len(PAM50_SUBTYPES),
        "er": 2,
        "pr": 2,
        "her2": 2,
    }
)

#: The three binary receptor tasks.
RECEPTOR_TASKS: Final[tuple[str, ...]] = ("er", "pr", "her2")

#: Human-readable class labels for the binary receptor tasks, indexed by class.
RECEPTOR_CLASS_LABELS: Final[tuple[str, ...]] = ("Negative", "Positive")

#: Human-readable class labels per classification task, indexed by class.
TASK_CLASS_LABELS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "subtype": PAM50_SUBTYPES,
        **{task: RECEPTOR_CLASS_LABELS for task in RECEPTOR_TASKS},
    }
)

#: Model output key carrying each classification task's logits.
TASK_LOGIT_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {task: f"{task}_logits" for task in CLASSIFICATION_TASKS}
)

#: Model output key carrying the survival risk score.
RISK_SCORE_KEY: Final[str] = "risk_score"
