"""Multi-task, mask-aware loss functions for BRCA prediction and survival.

Every task is masked independently: a patient contributes to a task's loss
only when that task's target is available (CLAUDE.md section 6). Two
properties matter and are enforced here:

1.  A masked-out row must not influence a task's loss *value*. Classification
    losses are averaged over the number of VALID rows, not the batch size, so
    a batch where half the HER2 labels are missing does not silently halve the
    HER2 loss.

2.  A task with zero valid rows in a batch must still produce a
    differentiable zero, so that ``backward()`` succeeds on batches where a
    task is entirely absent.

The Cox risk set is likewise restricted to the masked-in rows, so patients
excluded from the survival task (zero censored follow-up, unresolved
cross-source conflict) never appear in another patient's risk denominator.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.data.tasks import CLASSIFICATION_TASKS, IGNORE_INDEX, TASK_LOGIT_KEYS


def _zero_like_graph(reference: torch.Tensor) -> torch.Tensor:
    """Return a differentiable scalar zero attached to ``reference``'s graph.

    Multiplying by zero rather than returning ``torch.zeros(())`` keeps the
    autograd connection, so a batch in which a task has no valid rows still
    backpropagates. ``abs`` only normalises the signed zero, so that logs read
    ``0.0`` instead of ``-0.0``; the gradient contribution is zero either way.
    """
    return (reference.sum() * 0.0).abs()


def resolve_mask(
    targets: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Combine an explicit mask with the ignore-index sentinel.

    Args:
        targets: Integer class targets, possibly containing
            :data:`~src.data.tasks.IGNORE_INDEX`.
        mask: Optional boolean mask of valid rows.

    Returns:
        Boolean tensor of rows that are valid under both signals.
    """
    sentinel_valid = targets != IGNORE_INDEX
    if mask is None:
        return sentinel_valid
    return mask.bool().reshape(-1) & sentinel_valid


class FocalLoss(nn.Module):
    """Mask-aware focal loss for classification.

    Normalises over the number of valid rows so the loss magnitude does not
    depend on how many labels happen to be missing in a batch.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()

        if gamma < 0:
            raise ValueError("gamma must be >= 0.")

        self.gamma = gamma

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute focal loss over the valid rows only.

        Args:
            logits: Shape ``(batch, num_classes)``.
            targets: Shape ``(batch,)`` integer targets. Rows equal to
                :data:`~src.data.tasks.IGNORE_INDEX` are excluded.
            mask: Optional boolean mask of shape ``(batch,)``.

        Returns:
            Scalar loss. A differentiable zero when no row is valid.
        """
        targets = targets.long().reshape(-1)
        valid = resolve_mask(targets, mask)

        if not bool(valid.any()):
            return _zero_like_graph(logits)

        # Select before cross_entropy so sentinel targets never index a class.
        valid_logits = logits[valid]
        valid_targets = targets[valid]

        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction="none")

        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha.to(valid_logits.device).gather(0, valid_targets)
            focal_weight = focal_weight * alpha_t

        return (focal_weight * ce_loss).mean()


class CoxLoss(nn.Module):
    """Mask-aware negative Cox partial log-likelihood.

    Inputs are restricted to the masked-in rows and then sorted by descending
    survival duration, so the risk set of each event contains only patients
    who are actually part of the survival task.
    """

    def forward(
        self,
        risk_scores: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the negative Cox partial log-likelihood.

        Args:
            risk_scores: Shape ``(batch,)`` or ``(batch, 1)``.
            durations: Shape ``(batch,)`` follow-up times.
            events: Shape ``(batch,)`` event indicators (1 = death).
            mask: Optional boolean mask of patients eligible for the survival
                task.

        Returns:
            Scalar loss, normalised by the number of observed events. A
            differentiable zero when the batch has no eligible patients or no
            observed events.
        """
        risk = risk_scores.reshape(-1)
        durations = durations.reshape(-1)
        events = events.reshape(-1).float()

        if mask is not None:
            valid = mask.bool().reshape(-1)
            if not bool(valid.any()):
                return _zero_like_graph(risk_scores)
            risk = risk[valid]
            durations = durations[valid]
            events = events[valid]

        if risk.numel() == 0:
            return _zero_like_graph(risk_scores)

        n_events = events.sum()
        if n_events <= 0:
            # No observed events: the partial likelihood is undefined.
            return _zero_like_graph(risk_scores)

        order = torch.argsort(durations, descending=True)
        risk = risk[order]
        events = events[order]

        log_risk = torch.logcumsumexp(risk, dim=0)
        partial_log_likelihood = (risk - log_risk) * events

        return -partial_log_likelihood.sum() / n_events


class MultiTaskLoss(nn.Module):
    """Combined masked loss for subtype, receptor-status, and survival tasks.

    Tasks:
        - PAM50 molecular subtype: 5-class classification
        - ER status: binary classification
        - PR status: binary classification
        - HER2 status: binary classification
        - Overall survival: Cox proportional hazards

    The total objective is:

        L = w_subtype * L_subtype
            + w_er * L_er
            + w_pr * L_pr
            + w_her2 * L_her2
            + w_survival * L_survival

    where each term is computed over that task's valid rows only.
    """

    def __init__(
        self,
        subtype_weight: float = 1.0,
        er_weight: float = 1.0,
        pr_weight: float = 1.0,
        her2_weight: float = 1.0,
        survival_weight: float = 1.0,
        enable_survival: bool = True,
        focal_gamma: float = 2.0,
        subtype_class_weights: torch.Tensor | None = None,
        er_class_weights: torch.Tensor | None = None,
        pr_class_weights: torch.Tensor | None = None,
        her2_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.task_weights: dict[str, float] = {
            "subtype": subtype_weight,
            "er": er_weight,
            "pr": pr_weight,
            "her2": her2_weight,
        }
        self.survival_weight = survival_weight
        self.enable_survival = enable_survival

        self.task_losses = nn.ModuleDict(
            {
                "subtype": FocalLoss(alpha=subtype_class_weights, gamma=focal_gamma),
                "er": FocalLoss(alpha=er_class_weights, gamma=focal_gamma),
                "pr": FocalLoss(alpha=pr_class_weights, gamma=focal_gamma),
                "her2": FocalLoss(alpha=her2_class_weights, gamma=focal_gamma),
            }
        )

        self.cox_loss = CoxLoss()

    def forward(
        self,
        model_output: Mapping[str, torch.Tensor],
        labels: Mapping[str, torch.Tensor],
        masks: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all active task losses.

        Args:
            model_output: Model forward output containing ``<task>_logits``
                and, when survival is enabled, ``risk_score``.
            labels: Flat mapping with ``subtype``, ``er``, ``pr``, ``her2``,
                ``duration`` and ``event`` entries.
            masks: Optional per-task boolean masks. When omitted, masks are
                derived from the :data:`~src.data.tasks.IGNORE_INDEX` sentinel
                for classification tasks and all rows are used for survival.

        Returns:
            Mapping of task name to scalar loss, plus ``"total"``.

        Raises:
            KeyError: If a required logit key or label is absent.
        """
        masks = masks or {}
        result: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None

        for task in CLASSIFICATION_TASKS:
            logit_key = TASK_LOGIT_KEYS[task]
            if logit_key not in model_output:
                raise KeyError(f"Model output is missing '{logit_key}'.")
            if task not in labels:
                raise KeyError(f"Labels are missing task '{task}'.")

            task_loss = self.task_losses[task](
                model_output[logit_key], labels[task], masks.get(task)
            )
            result[task] = task_loss
            weighted = self.task_weights[task] * task_loss
            total = weighted if total is None else total + weighted

        if self.enable_survival:
            result["survival"] = self._survival_loss(model_output, labels, masks)
            total = total + self.survival_weight * result["survival"]

        if total is None:  # pragma: no cover - CLASSIFICATION_TASKS is non-empty
            raise ValueError("No tasks were configured for the loss.")

        result["total"] = total
        return result

    def _survival_loss(
        self,
        model_output: Mapping[str, torch.Tensor],
        labels: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the masked Cox loss, validating the required inputs."""
        if "risk_score" not in model_output:
            raise KeyError(
                "Survival is enabled but model output does not contain 'risk_score'."
            )
        for key in ("duration", "event"):
            if key not in labels:
                raise KeyError(f"Survival is enabled but labels lack '{key}'.")

        return self.cox_loss(
            model_output["risk_score"],
            labels["duration"],
            labels["event"],
            masks.get("survival"),
        )
