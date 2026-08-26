"""Explainability methods for the two-modality BRCA model.

Interpretability targets the two active modalities only: the 12-dimensional
clinical vector and the 50-gene PAM50 expression vector, plus the modality-level
cross-attention weights. Attributions are always taken with respect to one
named task head (``subtype``, ``er``, ``pr``, ``her2``), because the model has
no single ``logits`` output.

Attribution is conditioned on the *other* modality's real values rather than on
a single-modality forward pass, so the explanation reflects the fused model the
patient was actually scored by.

Optional libraries (shap, lime) are imported lazily inside each function so
importing this module does not require them.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.tasks import TASK_LOGIT_KEYS

#: Modalities that can be attributed in the active architecture.
ATTRIBUTABLE_MODALITIES = ("clinical", "genomics")


def _validate_modality(modality: str) -> None:
    """Reject modalities outside the active two-modality architecture."""
    if modality not in ATTRIBUTABLE_MODALITIES:
        raise ValueError(
            f"Unknown modality '{modality}'. Expected one of "
            f"{list(ATTRIBUTABLE_MODALITIES)}."
        )


def _task_logits(output: Mapping[str, torch.Tensor], task: str) -> torch.Tensor:
    """Return one task head's logits from a model output.

    Args:
        output: Model forward output.
        task: Classification task name.

    Returns:
        The logits tensor for ``task``.

    Raises:
        ValueError: If ``task`` is not a classification task.
        KeyError: If the model produced no logits for ``task``.
    """
    if task not in TASK_LOGIT_KEYS:
        raise ValueError(
            f"Unknown task '{task}'. Expected one of {sorted(TASK_LOGIT_KEYS)}."
        )

    key = TASK_LOGIT_KEYS[task]
    if key not in output:
        raise KeyError(f"Model output has no '{key}'. Found: {sorted(output)}.")

    return output[key]


def integrated_gradients(
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    modality: str = "clinical",
    task: str = "subtype",
    target_class: Optional[int] = None,
    n_steps: int = 50,
) -> np.ndarray:
    """Compute Integrated Gradients attribution for one modality.

    Approximates the path integral of gradients from a zero baseline to the
    input. The non-attributed modality is held at its real value across all
    interpolation steps, so the attribution explains the fused prediction.

    Args:
        model: Trained model.
        inputs: Modality tensors for a single patient, each shape ``(1, F)``.
            Must contain ``modality``.
        modality: Which modality to attribute (``"clinical"`` or
            ``"genomics"``).
        task: Task head to explain.
        target_class: Class index to explain. Uses the predicted class if
            ``None``.
        n_steps: Number of interpolation steps.

    Returns:
        Attribution array of shape ``(1, F)``.

    Raises:
        ValueError: If ``modality`` is unknown, absent from ``inputs``, or the
            tensor is not a single sample.
    """
    _validate_modality(modality)

    if modality not in inputs:
        raise ValueError(f"inputs has no '{modality}' tensor.")

    target = inputs[modality]
    if target.ndim != 2 or target.shape[0] != 1:
        raise ValueError(
            f"Expected a single sample of shape (1, F) for '{modality}', got "
            f"{tuple(target.shape)}."
        )

    model.eval()

    baseline = torch.zeros_like(target)
    alphas = torch.linspace(0.0, 1.0, n_steps, device=target.device).reshape(-1, 1)
    interpolated = (baseline + alphas * (target - baseline)).requires_grad_(True)

    forward_inputs = _tile_context(inputs, modality, interpolated, n_steps)

    logits = _task_logits(model(**forward_inputs), task)

    if target_class is None:
        target_class = int(logits[0].argmax().item())

    model.zero_grad(set_to_none=True)
    logits[:, target_class].sum().backward()

    if interpolated.grad is None:  # pragma: no cover - defensive
        raise RuntimeError("No gradient reached the interpolated input.")

    avg_grads = interpolated.grad.mean(dim=0, keepdim=True)
    return ((target - baseline) * avg_grads).detach().cpu().numpy()


def _tile_context(
    inputs: Mapping[str, torch.Tensor],
    modality: str,
    replacement: torch.Tensor,
    repeats: int,
) -> dict[str, torch.Tensor]:
    """Expand the non-attributed modalities to match a batched replacement."""
    forward_inputs: dict[str, torch.Tensor] = {modality: replacement}

    for name, tensor in inputs.items():
        if name == modality:
            continue
        _validate_modality(name)
        forward_inputs[name] = tensor.expand(repeats, *tensor.shape[1:])

    return forward_inputs


def attention_weights(
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Extract the modality-level cross-attention weights.

    :class:`~src.models.fusion.cross_modal_attention.CrossModalAttention` runs
    its inner ``MultiheadAttention`` with ``need_weights=False`` (the weights
    are not needed for training), so a plain forward hook cannot observe them.
    This function instead captures the query/key/value the inner module was
    called with and replays that one call with ``need_weights=True``. In eval
    mode the replay is deterministic, so the weights correspond exactly to the
    forward pass that produced the prediction.

    Args:
        model: Trained model exposing a ``cross_attention`` submodule.
        inputs: Modality tensors, each shape ``(batch, F)``.

    Returns:
        ``{"attention_weights": array}`` of shape
        ``(batch, heads, num_modalities, num_modalities)``, or ``{}`` when
        attention did not run (a single modality was supplied).
    """
    cross_modal = getattr(model, "cross_attention", None)
    inner = getattr(cross_modal, "cross_attention", None)

    if cross_modal is None or not isinstance(inner, nn.MultiheadAttention):
        return {}

    captured: list[tuple[torch.Tensor, ...]] = []

    def pre_hook(_: Any, args: tuple[Any, ...]) -> None:
        captured.append(tuple(a for a in args if isinstance(a, torch.Tensor)))

    handle = inner.register_forward_pre_hook(pre_hook)
    try:
        model.eval()
        with torch.no_grad():
            model(**dict(inputs))
    finally:
        handle.remove()

    if not captured or len(captured[0]) < 3:
        return {}

    query, key, value = captured[0][:3]
    with torch.no_grad():
        _, weights = inner(
            query,
            key,
            value,
            need_weights=True,
            average_attn_weights=False,
        )

    if weights is None:  # pragma: no cover - defensive
        return {}

    return {"attention_weights": weights.cpu().numpy()}


def shap_values(
    model: nn.Module,
    input_array: np.ndarray,
    background_array: np.ndarray,
    modality: str = "clinical",
    task: str = "subtype",
    target_class: int = 1,
    context: Optional[Mapping[str, torch.Tensor]] = None,
) -> np.ndarray:
    """Compute SHAP values for one modality using KernelExplainer.

    Args:
        model: Trained model.
        input_array: Samples to explain, shape ``(N, F)``.
        background_array: Background dataset, shape ``(K, F)``.
        modality: Which modality the arrays belong to.
        task: Task head to explain.
        target_class: Class index whose probability is explained.
        context: Fixed single-sample tensors for the other modality, shape
            ``(1, F)`` each. Required for a fused explanation.

    Returns:
        SHAP values array, shape ``(N, F)``.

    Raises:
        ImportError: If ``shap`` is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for shap_values(). Install it with: pip install shap"
        ) from exc

    predict_fn = _build_predict_fn(model, modality, task, context)

    def positive_class_fn(x: np.ndarray) -> np.ndarray:
        return predict_fn(x)[:, target_class]

    explainer = shap.KernelExplainer(positive_class_fn, background_array)
    return explainer.shap_values(input_array)


def lime_explanation(
    model: nn.Module,
    input_array: np.ndarray,
    feature_names: list[str],
    modality: str = "clinical",
    task: str = "subtype",
    target_class: int = 1,
    num_features: int = 10,
    num_samples: int = 500,
    context: Optional[Mapping[str, torch.Tensor]] = None,
) -> dict[str, float]:
    """Compute a LIME explanation for a single sample of one modality.

    Args:
        model: Trained model.
        input_array: Single sample, shape ``(F,)``.
        feature_names: Names for each feature dimension.
        modality: Which modality the array belongs to.
        task: Task head to explain.
        target_class: Class index to explain.
        num_features: Number of top features to report.
        num_samples: Number of perturbed samples LIME draws.
        context: Fixed single-sample tensors for the other modality.

    Returns:
        Mapping of feature name to LIME weight.

    Raises:
        ImportError: If ``lime`` is not installed.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError(
            "lime is required for lime_explanation(). Install it with: pip install lime"
        ) from exc

    predict_fn = _build_predict_fn(model, modality, task, context)

    explainer = LimeTabularExplainer(
        training_data=input_array.reshape(1, -1),
        feature_names=feature_names,
        mode="classification",
    )
    explanation = explainer.explain_instance(
        input_array,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=(target_class,),
    )
    return dict(explanation.as_list(label=target_class))


def _build_predict_fn(
    model: nn.Module,
    modality: str,
    task: str,
    context: Optional[Mapping[str, torch.Tensor]],
):
    """Build a numpy-in/numpy-out probability function for one task head.

    The returned callable is what the model-agnostic explainers (SHAP, LIME)
    perturb. Any ``context`` modality is tiled to the perturbation batch so the
    fused representation stays realistic.
    """
    _validate_modality(modality)
    device = next(model.parameters()).device
    model.eval()

    def predict_fn(x: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        forward_inputs = _tile_context(
            {modality: tensor, **{k: v.to(device) for k, v in (context or {}).items()}},
            modality,
            tensor,
            tensor.shape[0],
        )

        with torch.no_grad():
            logits = _task_logits(model(**forward_inputs), task)

        return torch.softmax(logits, dim=-1).cpu().numpy()

    return predict_fn
