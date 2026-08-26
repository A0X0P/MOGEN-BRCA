"""SHAP explainability for the frozen full multimodal TCGA-BRCA model.

Usage:
    uv run scripts/run_shap.py

Explains the already-trained checkpoint at
``results/breast/checkpoints/checkpoint_best.pt``. This script never trains,
never constructs an optimizer, and never writes inside ``results/breast/``.
All output goes to ``results/breast_explainability/``.

What is explained
-----------------
The model is multimodal, so SHAP is computed over a single **joint 62-feature
input space**:

    columns  0..49  the 50 PAM50 genes, in the checkpoint's gene order
    columns 50..61  the 12 clinical features, in CLINICAL_FEATURE_NAMES order

A thin wrapper splits that vector back into the two tensors the frozen model
expects and returns one scalar (or one vector, for PAM50) output. Attributing
both modalities in one shared space is what makes the genomic and clinical
contributions directly comparable on the same scale.

Explained outputs, stated exactly:

    er / pr / her2   the binary logit margin, logit(positive) - logit(negative),
                     which is a strictly increasing function of P(positive)
    subtype          the five PAM50 class logits, explained class by class
    survival         the DeepSurv risk score (the Cox linear predictor)

Method
------
``shap.GradientExplainer`` (expected gradients). It is PyTorch-native, needs no
op-by-op reimplementation of LayerNorm / MultiheadAttention / GELU, and
explains the actual forward pass rather than a surrogate. Background samples
come from the TRAIN partition; the explained samples are the TEST partition.

Interpretation limits
---------------------
SHAP values are model-attribution values: they describe how this trained model
distributes its output across its own inputs. They are not causal effects and
carry no biological claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # No display in this environment; write files only.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_eval import build_dataloader, rebuild_split  # noqa: E402
from scripts.run_train import load_merged_config  # noqa: E402
from src.data.datasets.multimodal_dataset import MultimodalDataset  # noqa: E402
from src.data.pam50 import PAM50_SUBTYPES  # noqa: E402
from src.data.schema.clinical import CLINICAL_FEATURE_NAMES  # noqa: E402
from src.data.schema.patient import BrcaTargets, Patient  # noqa: E402
from src.evaluation.evaluator import evaluate  # noqa: E402
from src.inference.predict import InferenceArtifacts, load_model  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

CHECKPOINT = REPO_ROOT / "results/breast/checkpoints/checkpoint_best.pt"
TRAIN_CONFIG = REPO_ROOT / "configs/breast/train.yaml"
RECORDED_METRICS = REPO_ROOT / "results/breast/test_metrics.json"
OUTPUT_DIR = REPO_ROOT / "results/breast_explainability"

#: Modality label written into the feature-importance table.
GENOMIC_MODALITY = "genomic"
CLINICAL_MODALITY = "clinical"

#: Bar colours per modality, so every figure separates the two visually.
MODALITY_COLOR = {GENOMIC_MODALITY: "#2f6f9f", CLINICAL_MODALITY: "#c8792b"}

#: Features shown in a bar/beeswarm panel.
TOP_N_DISPLAY = 20

#: Features listed in the per-task top-feature table.
TOP_N_TABLE = 20

#: |Pearson r| between a feature's value and its SHAP value below which the
#: direction of contribution is reported as mixed rather than monotone. A low
#: correlation with a high mean |SHAP| means the feature matters but pushes
#: different patients in different directions, which is worth naming rather than
#: forcing into a single sign.
DIRECTION_CORRELATION_CUTOFF = 0.1

#: How the representative patients shown in the individual explanations are
#: chosen. The rule uses model predictions only, never labels, so no test label
#: influences which patients are displayed. Each patient's true label is
#: reported alongside for context.
REPRESENTATIVE_SELECTION = (
    ("highest_output", "patient with the highest explained output"),
    ("median_output", "patient at the median explained output"),
    ("lowest_output", "patient with the lowest explained output"),
)


@dataclass(frozen=True)
class ExplainedOutput:
    """One model output to attribute.

    Attributes:
        task: Task name used in filenames and the importance table.
        title: Human-readable description of the explained quantity.
        select: Maps the model's output dict to the tensor being explained,
            shaped ``(batch, n_outputs)``.
        output_names: One label per column of the selected tensor.
    """

    task: str
    title: str
    select: Callable[[dict[str, torch.Tensor]], torch.Tensor]
    output_names: tuple[str, ...]


def _margin(logits: torch.Tensor) -> torch.Tensor:
    """Binary logit margin, positive minus negative class."""
    return (logits[:, 1] - logits[:, 0]).unsqueeze(1)


EXPLAINED_OUTPUTS: tuple[ExplainedOutput, ...] = (
    ExplainedOutput(
        task="er",
        title="ER logit margin, logit(positive) - logit(negative)",
        select=lambda out: _margin(out["er_logits"]),
        output_names=("er_margin",),
    ),
    ExplainedOutput(
        task="pr",
        title="PR logit margin, logit(positive) - logit(negative)",
        select=lambda out: _margin(out["pr_logits"]),
        output_names=("pr_margin",),
    ),
    ExplainedOutput(
        task="her2",
        title="HER2 logit margin, logit(positive) - logit(negative)",
        select=lambda out: _margin(out["her2_logits"]),
        output_names=("her2_margin",),
    ),
    ExplainedOutput(
        task="pam50",
        title="PAM50 class logits (five classes, explained separately)",
        select=lambda out: out["subtype_logits"],
        output_names=PAM50_SUBTYPES,
    ),
    ExplainedOutput(
        task="survival",
        title="DeepSurv risk score (Cox linear predictor)",
        select=lambda out: out["risk_score"].reshape(out["risk_score"].shape[0], -1),
        output_names=("risk_score",),
    ),
)


class JointInputModel(nn.Module):
    """Adapts the frozen two-input model to a single joint feature vector.

    The genomic block occupies the leading ``n_genes`` columns and the clinical
    block the remainder, matching the concatenation order of
    :func:`build_feature_matrix`. Splitting here — rather than reimplementing
    any part of the model — keeps the explained function identical to the
    deployed forward pass.

    Args:
        model: The loaded, frozen multimodal model.
        output: The output to expose.
        n_genes: Width of the genomic block.
    """

    def __init__(self, model: nn.Module, output: ExplainedOutput, n_genes: int) -> None:
        super().__init__()
        self.model = model
        self._output = output
        self._n_genes = n_genes

    def forward(self, joint: torch.Tensor) -> torch.Tensor:
        """Split the joint vector, run the frozen model, select the output."""
        if joint.ndim != 2:
            raise ValueError(f"Expected a 2-D (batch, features) input, got {tuple(joint.shape)}.")

        genomics = joint[:, : self._n_genes]
        clinical = joint[:, self._n_genes :]
        return self._output.select(self.model(clinical=clinical, genomics=genomics))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
        help="Number of TRAIN patients used as the SHAP background distribution.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=200,
        help="Expected-gradients samples per explained patient.",
    )
    parser.add_argument(
        "--explain-limit",
        type=int,
        default=None,
        help="Explain only the first N test patients (smoke runs). Default: all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for background selection and expected-gradients sampling.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Feature space
# ----------------------------------------------------------------------------


def build_feature_matrix(
    patients: Sequence[Patient], artifacts: InferenceArtifacts
) -> tuple[torch.Tensor, list[str]]:
    """Encode patients into the joint ``(n_patients, 62)`` feature matrix.

    Uses the same :class:`MultimodalDataset` encoding path the model was
    trained and evaluated with, and the preprocessing statistics carried inside
    the checkpoint, so no statistic is refitted here. Only the two feature
    tensors are read; targets, masks, and survival times are ignored.

    Args:
        patients: Patients to encode.
        artifacts: Loaded checkpoint artifacts supplying the train-fold
            preprocessing statistics and gene order.

    Returns:
        The feature matrix (genomic block first) and the patient identifiers in
        matrix row order.
    """
    dataset = MultimodalDataset(
        patients=list(patients),
        normalization_stats=artifacts.normalization_stats,
        gene_standardization=artifacts.gene_standardization,
        gene_order=artifacts.gene_order,
    )
    rows = [dataset[index] for index in range(len(dataset))]

    genomics = torch.stack([row["genomics"]["features"] for row in rows])
    clinical = torch.stack([row["clinical"]["features"] for row in rows])
    return torch.cat([genomics, clinical], dim=1), dataset.patient_ids


def feature_names(artifacts: InferenceArtifacts) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the joint feature names and their per-column modality labels."""
    names = tuple(artifacts.gene_order) + tuple(CLINICAL_FEATURE_NAMES)
    modalities = tuple(
        [GENOMIC_MODALITY] * len(artifacts.gene_order)
        + [CLINICAL_MODALITY] * len(CLINICAL_FEATURE_NAMES)
    )
    return names, modalities


def select_background(
    patients: Sequence[Patient], size: int, seed: int
) -> list[Patient]:
    """Deterministically sample the background patients from the TRAIN split."""
    if size >= len(patients):
        return list(patients)

    ordered = sorted(patients, key=lambda patient: patient.patient_id)
    indices = np.random.default_rng(seed).choice(len(ordered), size=size, replace=False)
    return [ordered[int(index)] for index in sorted(indices)]


# ----------------------------------------------------------------------------
# SHAP
# ----------------------------------------------------------------------------


def as_output_list(values: Any, n_outputs: int) -> list[np.ndarray]:
    """Normalise SHAP's return value to one ``(n_samples, n_features)`` array per output.

    Raises:
        ValueError: If the returned shape cannot be reconciled with
            ``n_outputs``.
    """
    if isinstance(values, list):
        arrays = [np.asarray(value) for value in values]
    else:
        array = np.asarray(values)
        if array.ndim == 3:
            arrays = [array[..., index] for index in range(array.shape[-1])]
        elif array.ndim == 2:
            arrays = [array]
        else:
            raise ValueError(f"Unexpected SHAP value shape {array.shape}.")

    if len(arrays) != n_outputs:
        raise ValueError(
            f"SHAP returned {len(arrays)} output(s), expected {n_outputs}."
        )
    return arrays


def explain_output(
    model: nn.Module,
    output: ExplainedOutput,
    background: torch.Tensor,
    explain: torch.Tensor,
    n_genes: int,
    nsamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute SHAP values for one model output.

    Args:
        model: The frozen multimodal model.
        output: Which output to explain.
        background: Background feature matrix from the TRAIN partition.
        explain: Feature matrix of the patients being explained.
        n_genes: Width of the genomic block.
        nsamples: Expected-gradients samples per explained patient.
        seed: Seed for the explainer's internal sampling.

    Returns:
        Per-output SHAP arrays plus the base value, prediction, and additivity
        diagnostic used by the sanity checks.
    """
    import shap

    wrapper = JointInputModel(model, output, n_genes).eval()

    with torch.no_grad():
        base_values = wrapper(background).mean(dim=0)
        predictions = wrapper(explain)

    torch.manual_seed(seed)
    explainer = shap.GradientExplainer(wrapper, background)
    arrays = as_output_list(
        explainer.shap_values(explain, nsamples=nsamples, rseed=seed),
        len(output.output_names),
    )

    gaps: list[dict[str, float]] = []
    for index, array in enumerate(arrays):
        attributed = array.sum(axis=1)
        expected = (predictions[:, index] - base_values[index]).numpy()
        residual = np.abs(attributed - expected)
        gaps.append(
            {
                "mean_abs_additivity_gap": float(residual.mean()),
                "max_abs_additivity_gap": float(residual.max()),
                "std_of_explained_deviation": float(expected.std()),
            }
        )

    logger.info(
        "%s: SHAP computed for %d output(s) over %d patients (nsamples=%d).",
        output.task,
        len(arrays),
        explain.shape[0],
        nsamples,
    )

    return {
        "arrays": arrays,
        "base_values": base_values.numpy().tolist(),
        "predictions": predictions.numpy(),
        "additivity": gaps,
    }


def mean_abs_importance(array: np.ndarray) -> np.ndarray:
    """Mean absolute SHAP value per feature."""
    return np.abs(array).mean(axis=0)


def mean_signed_shap(array: np.ndarray) -> np.ndarray:
    """Mean *signed* SHAP value per feature.

    Distinguishes a feature that consistently pushes the output one way from one
    that pushes different patients in opposite directions: the latter has a large
    mean absolute value and a near-zero mean signed value.
    """
    return array.mean(axis=0)


def feature_value_correlation(array: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Pearson correlation between each feature's value and its SHAP value.

    This is the direction of contribution: a positive correlation means a higher
    value of that feature pushes the explained output up across the explained
    cohort. Constant features (a one-hot level nobody in the cohort has) have no
    defined correlation and yield ``nan``.

    Args:
        array: SHAP values, shape ``(n_patients, n_features)``.
        features: Feature values, same shape.

    Returns:
        Correlation per feature, shape ``(n_features,)``, with ``nan`` where the
        feature or its attribution is constant.
    """
    correlations = np.full(array.shape[1], np.nan)
    for index in range(array.shape[1]):
        values, shap_values = features[:, index], array[:, index]
        if np.std(values) == 0.0 or np.std(shap_values) == 0.0:
            continue
        correlations[index] = float(np.corrcoef(values, shap_values)[0, 1])
    return correlations


def direction_label(correlation: float) -> str:
    """Name the direction of contribution implied by a feature/SHAP correlation."""
    if not np.isfinite(correlation):
        return "undefined (feature constant across explained patients)"
    if correlation >= DIRECTION_CORRELATION_CUTOFF:
        return "higher feature value increases the output"
    if correlation <= -DIRECTION_CORRELATION_CUTOFF:
        return "higher feature value decreases the output"
    return "mixed (no consistent direction across patients)"


def representative_indices(predictions: np.ndarray) -> dict[str, int]:
    """Pick the highest, median and lowest patient by explained output.

    Selection depends only on the model's own output, never on a label, so
    displaying these patients cannot leak test labels into any decision. The
    median is the patient at the middle rank, not an interpolated value, so a
    real patient is always shown.

    Args:
        predictions: Explained output per patient, shape ``(n_patients,)``.

    Returns:
        Mapping from selection key to row index.
    """
    order = np.argsort(predictions)
    return {
        "highest_output": int(order[-1]),
        "median_output": int(order[len(order) // 2]),
        "lowest_output": int(order[0]),
    }


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def plot_global_bar(
    importance: np.ndarray,
    names: Sequence[str],
    modalities: Sequence[str],
    title: str,
    path: Path,
) -> None:
    """Two-panel global figure: top features by modality, and modality totals.

    The left panel ranks individual features and colours each bar by its
    modality; the right panel sums mean |SHAP| within each modality, which is
    the figure-level answer to "how much of this prediction comes from gene
    expression versus clinical variables".
    """
    order = np.argsort(importance)[::-1][:TOP_N_DISPLAY][::-1]
    labels = [names[index] for index in order]
    colors = [MODALITY_COLOR[modalities[index]] for index in order]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 7.2), gridspec_kw={"width_ratios": [3, 1]}
    )

    left.barh(range(len(order)), importance[order], color=colors)
    left.set_yticks(range(len(order)))
    left.set_yticklabels(labels, fontsize=9)
    left.set_xlabel("mean |SHAP value|")
    left.set_title(f"Top {len(order)} features", fontsize=10)
    left.grid(axis="x", alpha=0.3)

    totals = {
        modality: float(
            importance[[i for i, m in enumerate(modalities) if m == modality]].sum()
        )
        for modality in (GENOMIC_MODALITY, CLINICAL_MODALITY)
    }
    total = sum(totals.values()) or 1.0
    keys = list(totals)
    right.bar(keys, [totals[k] for k in keys], color=[MODALITY_COLOR[k] for k in keys])
    for index, key in enumerate(keys):
        right.text(
            index,
            totals[key],
            f"{100.0 * totals[key] / total:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    right.set_ylabel("summed mean |SHAP value|")
    right.set_title("Modality contribution", fontsize=10)
    right.grid(axis="y", alpha=0.3)

    figure.suptitle(f"{title}\nSHAP attribution (not causal effect)", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("Wrote %s.", path.name)


def plot_beeswarm(
    array: np.ndarray,
    features: np.ndarray,
    names: Sequence[str],
    title: str,
    path: Path,
) -> None:
    """Standard SHAP beeswarm for one output."""
    import shap

    plt.figure()
    shap.summary_plot(
        array,
        features=features,
        feature_names=list(names),
        max_display=TOP_N_DISPLAY,
        plot_type="dot",
        show=False,
    )
    plt.title(f"{title}\nSHAP attribution (not causal effect)", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Wrote %s.", path.name)


def plot_pam50_class_bars(
    arrays: Sequence[np.ndarray],
    features: np.ndarray,
    names: Sequence[str],
    path: Path,
) -> None:
    """Stacked per-class importance bar for the five PAM50 classes."""
    import shap

    plt.figure()
    shap.summary_plot(
        list(arrays),
        features=features,
        feature_names=list(names),
        class_names=list(PAM50_SUBTYPES),
        max_display=TOP_N_DISPLAY,
        plot_type="bar",
        show=False,
    )
    plt.title(
        "PAM50 class-specific SHAP importance\nSHAP attribution (not causal effect)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Wrote %s.", path.name)


def plot_patient_waterfall(
    array: np.ndarray,
    features: np.ndarray,
    names: Sequence[str],
    modalities: Sequence[str],
    row: int,
    base_value: float,
    prediction: float,
    title: str,
    path: Path,
) -> None:
    """Per-patient attribution figure for one explained output.

    Bars are the individual SHAP values for this patient's top features, ordered
    by magnitude and coloured by modality, so the genomic and clinical
    contributions to a single prediction are visually separable. The base value
    and the model's output for this patient are annotated, because an attribution
    is only interpretable relative to the background it is measured against.

    Args:
        array: SHAP values for the explained output, shape ``(n_patients, n_features)``.
        features: Feature values, same shape.
        names: Feature names.
        modalities: Per-feature modality labels.
        row: Row index of the patient to plot.
        base_value: Mean explained output over the background patients.
        prediction: The model's explained output for this patient.
        title: Figure title.
        path: Destination PNG path.
    """
    values = array[row]
    order = np.argsort(np.abs(values))[::-1][:TOP_N_DISPLAY][::-1]

    labels = [f"{names[index]} = {features[row, index]:+.2f}" for index in order]
    colors = [MODALITY_COLOR[modalities[index]] for index in order]

    figure, axis = plt.subplots(figsize=(9.5, 7.0))
    axis.barh(range(len(order)), values[order], color=colors)
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("SHAP value (contribution to this patient's output)")
    axis.grid(axis="x", alpha=0.3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODALITY_COLOR[modality])
        for modality in (GENOMIC_MODALITY, CLINICAL_MODALITY)
    ]
    axis.legend(handles, [GENOMIC_MODALITY, CLINICAL_MODALITY], loc="lower right", fontsize=8)

    figure.suptitle(
        f"{title}\nbase value {base_value:+.3f} -> model output {prediction:+.3f}"
        f"   (sum of all 62 SHAP values {values.sum():+.3f})"
        "\nSHAP attribution (not causal effect)",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("Wrote %s.", path.name)


def slug(text: str) -> str:
    """Filename-safe lowercase slug."""
    return text.lower().replace(" ", "_").replace("-", "_")


# ----------------------------------------------------------------------------
# Sanity checks (Objective 5)
# ----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_parameters(model: nn.Module) -> str:
    """SHA-256 over every parameter tensor, in a fixed name order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten a nested metrics dict to ``{dotted.key: float}``."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten_numeric(value, f"{prefix}{key}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = float(obj)
    return out


def verify_recorded_metrics(
    artifacts: InferenceArtifacts, test_patients: Sequence[Patient]
) -> dict[str, Any]:
    """Re-evaluate the loaded checkpoint and compare to the recorded report.

    Confirms the weights being explained are the same weights that produced the
    frozen test metrics. Test labels are used here only to recompute the
    already-published metrics, not to influence SHAP.
    """
    loader = build_dataloader(list(test_patients), artifacts, batch_size=32)
    fresh = flatten_numeric(
        {k: v for k, v in evaluate(
            artifacts.model, loader, device=artifacts.device
        ).to_dict().items() if k != "n_samples"}
    )
    recorded = flatten_numeric(
        {
            k: v
            for k, v in json.loads(RECORDED_METRICS.read_text(encoding="utf-8")).items()
            if k not in ("partition", "checkpoint", "n_samples")
        }
    )

    if set(fresh) != set(recorded):
        raise ValueError(f"Metric keys differ: {sorted(set(fresh) ^ set(recorded))}.")

    worst_key = max(fresh, key=lambda key: abs(fresh[key] - recorded[key]))
    worst = abs(fresh[worst_key] - recorded[worst_key])

    return {
        "metrics_compared": len(fresh),
        "max_abs_difference": worst,
        "worst_metric": worst_key,
        "reproduced": worst < 1e-9,
        "source": str(RECORDED_METRICS.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


def verify_label_independence(
    patients: Sequence[Patient], artifacts: InferenceArtifacts, reference: torch.Tensor
) -> dict[str, Any]:
    """Prove the SHAP input matrix does not depend on any target label.

    Re-encodes the same patients with every target blanked. If the resulting
    matrix is bit-identical, no label can have entered the explained features.
    """
    blanked = [
        patient.model_copy(update={"targets": BrcaTargets()}) for patient in patients
    ]
    matrix, _ = build_feature_matrix(blanked, artifacts)
    identical = bool(torch.equal(matrix, reference))

    return {
        "method": "re-encoded the same patients with all targets blanked",
        "matrices_bit_identical": identical,
        "max_abs_difference": float((matrix - reference).abs().max()),
        "passed": identical,
    }


def verify_input_ordering(
    model: nn.Module, joint: torch.Tensor, n_genes: int
) -> dict[str, Any]:
    """Prove the joint 62-column layout reproduces the model's own two inputs.

    Two pieces of evidence. First, the wrapper's output must equal a direct call
    that passes the two blocks separately — if the split point or column order
    were wrong, the outputs would diverge. Second, perturbing each block alone
    must move the output, which shows both blocks are genuinely wired to the
    forward pass rather than one being silently ignored.
    """
    sample = joint[: min(16, joint.shape[0])]
    wrapper = JointInputModel(model, EXPLAINED_OUTPUTS[0], n_genes).eval()

    genomic_perturbed = sample.clone()
    genomic_perturbed[:, :n_genes] += 1.0
    clinical_perturbed = sample.clone()
    clinical_perturbed[:, n_genes:] += 1.0

    with torch.no_grad():
        through_wrapper = wrapper(sample)
        direct = _margin(
            model(clinical=sample[:, n_genes:], genomics=sample[:, :n_genes])[
                "er_logits"
            ]
        )
        genomic_shift = float((wrapper(genomic_perturbed) - through_wrapper).abs().max())
        clinical_shift = float(
            (wrapper(clinical_perturbed) - through_wrapper).abs().max()
        )

    difference = float((through_wrapper - direct).abs().max())
    return {
        "wrapper_matches_direct_call": difference == 0.0,
        "max_abs_difference": difference,
        "genomic_block_influences_output": genomic_shift > 1e-6,
        "clinical_block_influences_output": clinical_shift > 1e-6,
        "genomic_perturbation_output_shift": genomic_shift,
        "clinical_perturbation_output_shift": clinical_shift,
        "layout": f"columns 0..{n_genes - 1} genomic, {n_genes}..{joint.shape[1] - 1} clinical",
    }


def run_sanity_checks(
    artifacts: InferenceArtifacts,
    results: dict[str, dict[str, Any]],
    names: Sequence[str],
    modalities: Sequence[str],
    joint_test: torch.Tensor,
    checkpoint_hash_before: str,
    parameter_hash_before: str,
    metric_check: dict[str, Any],
    label_check: dict[str, Any],
    ordering_check: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run and record the ten required SHAP sanity checks.

    Raises:
        RuntimeError: If any check fails.
    """
    model = artifacts.model
    n_genes = len(artifacts.gene_order)
    n_features = joint_test.shape[1]

    finite = all(
        bool(np.isfinite(array).all())
        for result in results.values()
        for array in result["arrays"]
    )
    widths = {
        array.shape[1]
        for result in results.values()
        for array in result["arrays"]
    }
    grads_populated = sum(
        1 for parameter in model.parameters() if parameter.grad is not None
    )

    checks = {
        "1_model_in_eval_mode": {
            "passed": not model.training,
            "training_flag_after_shap": model.training,
            "note": "Dropout and any train-time behaviour are disabled.",
        },
        "2_no_optimizer_step": {
            "passed": parameter_hash_before == sha256_parameters(model),
            "optimizer_constructed": False,
            "parameters_with_populated_grad": grads_populated,
            "note": (
                "No torch.optim object exists in this script. Expected "
                "gradients use torch.autograd.grad, which does not accumulate "
                "into .grad or modify weights."
            ),
        },
        "3_checkpoint_weights_unchanged": {
            "passed": (
                sha256_file(CHECKPOINT) == checkpoint_hash_before
                and sha256_parameters(model) == parameter_hash_before
            ),
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": sha256_file(CHECKPOINT),
            "parameter_sha256_before": parameter_hash_before,
            "parameter_sha256_after": sha256_parameters(model),
        },
        "4_predictions_match_recorded_metrics": {
            "passed": bool(metric_check["reproduced"]),
            **metric_check,
        },
        "5_feature_order_matches_model_input": {
            "passed": (
                tuple(names[:n_genes]) == tuple(artifacts.gene_order)
                and tuple(names[n_genes:]) == tuple(CLINICAL_FEATURE_NAMES)
                and bool(ordering_check["wrapper_matches_direct_call"])
                and bool(ordering_check["genomic_block_influences_output"])
                and bool(ordering_check["clinical_block_influences_output"])
            ),
            "gene_order_source": "checkpoint config data.gene_order",
            "clinical_order_source": "src.data.schema.clinical.CLINICAL_FEATURE_NAMES",
            **ordering_check,
        },
        "6_no_test_labels_used_for_shap": {
            "passed": bool(label_check["passed"]),
            **label_check,
        },
        "7_feature_names_match_model_inputs": {
            "passed": (
                len(names) == n_features
                and len(modalities) == n_features
                and n_genes == int(artifacts.config["model"]["genomics_input_dim"])
                and len(CLINICAL_FEATURE_NAMES)
                == int(artifacts.config["model"]["clinical_input_dim"])
            ),
            "n_names": len(names),
            "genomics_input_dim": int(artifacts.config["model"]["genomics_input_dim"]),
            "clinical_input_dim": int(artifacts.config["model"]["clinical_input_dim"]),
        },
        "8_shap_values_finite": {
            "passed": finite,
            "arrays_checked": sum(len(r["arrays"]) for r in results.values()),
        },
        "9_shap_feature_count_matches_input_dim": {
            "passed": widths == {n_features},
            "shap_feature_widths": sorted(widths),
            "model_input_dim": n_features,
        },
        "10_environment_recorded": {
            "passed": True,
            "environment": environment_record(),
            "explainer": {
                "name": "shap.GradientExplainer",
                "algorithm": "expected gradients",
                "nsamples_per_patient": args.nsamples,
                "background_size": args.background_size,
                "background_partition": "train",
                "explained_partition": "test",
                "seed": args.seed,
            },
        },
    }

    failed = [name for name, check in checks.items() if not check["passed"]]
    if failed:
        raise RuntimeError(f"SHAP sanity checks failed: {failed}.")

    logger.info("All %d SHAP sanity checks passed.", len(checks))
    return checks


def environment_record() -> dict[str, Any]:
    """Library versions and platform facts for the reproducibility record."""
    import shap
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "shap": shap.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
    }


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------


def importance_rows(
    results: dict[str, dict[str, Any]],
    names: Sequence[str],
    modalities: Sequence[str],
    features: np.ndarray,
) -> list[dict[str, Any]]:
    """Build the ``shap_feature_importance.csv`` rows, ranked within each task.

    PAM50 contributes one row set per class plus a ``pam50`` row set averaged
    over the five classes.

    Each row carries the mean absolute SHAP value (magnitude, which sets the
    rank), the mean signed value, and the feature/SHAP correlation with its
    direction label. The class-averaged PAM50 rows have no single signed value —
    averaging magnitudes across five class logits discards the sign — so their
    signed and direction columns are left empty rather than filled with a
    misleading number.

    Args:
        results: Per-task SHAP results.
        names: Feature names.
        modalities: Per-feature modality labels.
        features: The explained patients' feature matrix.

    Returns:
        One row per (task, feature) pair.
    """
    rows: list[dict[str, Any]] = []

    for output in EXPLAINED_OUTPUTS:
        result = results[output.task]

        labelled: list[tuple[str, np.ndarray, Optional[np.ndarray]]] = []
        per_output = [mean_abs_importance(array) for array in result["arrays"]]
        if len(per_output) > 1:
            labelled.append((output.task, np.mean(per_output, axis=0), None))
            labelled.extend(
                (f"{output.task}_{slug(name)}", per_output[index], result["arrays"][index])
                for index, name in enumerate(output.output_names)
            )
        else:
            labelled.append((output.task, per_output[0], result["arrays"][0]))

        for task_label, importance, array in labelled:
            signed = None if array is None else mean_signed_shap(array)
            correlation = (
                None if array is None else feature_value_correlation(array, features)
            )
            order = np.argsort(importance)[::-1]
            for rank, index in enumerate(order, start=1):
                rows.append(
                    {
                        "task": task_label,
                        "modality": modalities[index],
                        "feature": names[index],
                        "mean_abs_shap": float(importance[index]),
                        "rank": rank,
                        "mean_shap": "" if signed is None else float(signed[index]),
                        "feature_shap_correlation": (
                            ""
                            if correlation is None or not np.isfinite(correlation[index])
                            else float(correlation[index])
                        ),
                        "direction": (
                            ""
                            if correlation is None
                            else direction_label(correlation[index])
                        ),
                    }
                )

    return rows


def top_feature_rows(
    results: dict[str, dict[str, Any]],
    names: Sequence[str],
    modalities: Sequence[str],
    features: np.ndarray,
) -> list[dict[str, Any]]:
    """Top-``TOP_N_TABLE`` features per explained output, with direction.

    A separate, smaller table from the full ranking, so the headline features for
    each of the ten explained outputs can be read without filtering 620 rows.
    Genomic and clinical features are labelled, and the modality is carried on
    every row, so the two blocks stay distinguishable in the reporting.
    """
    rows: list[dict[str, Any]] = []

    for output in EXPLAINED_OUTPUTS:
        result = results[output.task]
        for index, output_name in enumerate(output.output_names):
            array = result["arrays"][index]
            importance = mean_abs_importance(array)
            signed = mean_signed_shap(array)
            correlation = feature_value_correlation(array, features)
            order = np.argsort(importance)[::-1][:TOP_N_TABLE]

            for rank, feature_index in enumerate(order, start=1):
                rows.append(
                    {
                        "task": output.task,
                        "explained_output": output_name,
                        "rank": rank,
                        "modality": modalities[feature_index],
                        "feature": names[feature_index],
                        "mean_abs_shap": float(importance[feature_index]),
                        "mean_shap": float(signed[feature_index]),
                        "feature_shap_correlation": (
                            float(correlation[feature_index])
                            if np.isfinite(correlation[feature_index])
                            else ""
                        ),
                        "direction": direction_label(correlation[feature_index]),
                    }
                )

    return rows


def individual_rows(
    results: dict[str, dict[str, Any]],
    names: Sequence[str],
    modalities: Sequence[str],
    features: np.ndarray,
    explained_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Per-patient attributions for the representative patients.

    For every explained output, three patients are selected by model output alone
    (highest, median, lowest) and their top-``TOP_N_TABLE`` feature attributions
    are written out. The row carries the feature's encoded value, so a reader can
    see *what* the model saw and not only how much it mattered.
    """
    rows: list[dict[str, Any]] = []

    for output in EXPLAINED_OUTPUTS:
        result = results[output.task]
        for index, output_name in enumerate(output.output_names):
            array = result["arrays"][index]
            predictions = result["predictions"][:, index]
            base_value = float(result["base_values"][index])

            for key, patient_row in representative_indices(predictions).items():
                values = array[patient_row]
                order = np.argsort(np.abs(values))[::-1][:TOP_N_TABLE]
                for rank, feature_index in enumerate(order, start=1):
                    rows.append(
                        {
                            "task": output.task,
                            "explained_output": output_name,
                            "selection": key,
                            "patient_id": explained_ids[patient_row],
                            "base_value": base_value,
                            "model_output": float(predictions[patient_row]),
                            "sum_all_shap_values": float(values.sum()),
                            "rank": rank,
                            "modality": modalities[feature_index],
                            "feature": names[feature_index],
                            "feature_value_standardised": float(
                                features[patient_row, feature_index]
                            ),
                            "shap_value": float(values[feature_index]),
                            "pushes_output": (
                                "up" if values[feature_index] > 0 else "down"
                            ),
                        }
                    )

    return rows


def modality_split(
    importance: np.ndarray, modalities: Sequence[str]
) -> dict[str, Any]:
    """Summed mean |SHAP| per modality, with each modality's share."""
    totals = {
        modality: float(
            importance[[i for i, m in enumerate(modalities) if m == modality]].sum()
        )
        for modality in (GENOMIC_MODALITY, CLINICAL_MODALITY)
    }
    total = sum(totals.values())
    return {
        "summed_mean_abs_shap": totals,
        "share": {
            modality: (value / total if total else None)
            for modality, value in totals.items()
        },
        "note": (
            "Shares compare the two modalities within one shared 62-feature "
            "attribution space. The genomic block has 50 features and the "
            "clinical block 12, so a larger genomic total partly reflects that "
            "width; the per-feature means in the table are the fairer "
            "feature-level comparison."
        ),
    }


def build_summary(
    results: dict[str, dict[str, Any]],
    names: Sequence[str],
    modalities: Sequence[str],
    features: np.ndarray,
    checks: dict[str, Any],
    artifacts: InferenceArtifacts,
    checkpoint_epoch: int,
    background_ids: Sequence[str],
    explained_ids: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Assemble ``shap_summary.json``."""
    tasks: dict[str, Any] = {}

    for output in EXPLAINED_OUTPUTS:
        result = results[output.task]
        per_output = [mean_abs_importance(array) for array in result["arrays"]]
        pooled = np.mean(per_output, axis=0)

        outputs = {}
        for index, output_name in enumerate(output.output_names):
            array = result["arrays"][index]
            signed = mean_signed_shap(array)
            correlation = feature_value_correlation(array, features)
            predictions = result["predictions"][:, index]
            order = np.argsort(per_output[index])[::-1][:TOP_N_TABLE]

            outputs[output_name] = {
                "base_value": result["base_values"][index],
                "additivity": result["additivity"][index],
                "explained_output_range": {
                    "min": float(predictions.min()),
                    "median": float(np.median(predictions)),
                    "max": float(predictions.max()),
                },
                "representative_patients": {
                    key: {
                        "patient_id": explained_ids[row],
                        "model_output": float(predictions[row]),
                        "sum_of_shap_values": float(array[row].sum()),
                        "selection_rule": description,
                    }
                    for (key, description), row in zip(
                        REPRESENTATIVE_SELECTION,
                        [
                            representative_indices(predictions)[key]
                            for key, _ in REPRESENTATIVE_SELECTION
                        ],
                    )
                },
                f"top_{TOP_N_TABLE}_features": [
                    {
                        "feature": names[i],
                        "modality": modalities[i],
                        "mean_abs_shap": float(per_output[index][i]),
                        "mean_shap": float(signed[i]),
                        "feature_shap_correlation": (
                            float(correlation[i]) if np.isfinite(correlation[i]) else None
                        ),
                        "direction": direction_label(correlation[i]),
                    }
                    for i in order
                ],
            }

        tasks[output.task] = {
            "explained_quantity": output.title,
            "n_outputs": len(output.output_names),
            "modality_contribution": modality_split(pooled, modalities),
            "outputs": outputs,
        }

    return {
        "purpose": (
            "Attribute the frozen full multimodal model's outputs to its 62 "
            "input features (50 PAM50 genes + 12 clinical features)."
        ),
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": checks["3_checkpoint_weights_unchanged"][
                "checkpoint_sha256_after"
            ],
            "epoch": checkpoint_epoch,
            "total_parameters": sum(
                parameter.numel() for parameter in artifacts.model.parameters()
            ),
            "retrained": False,
            "weights_modified": False,
        },
        "feature_space": {
            "n_features": len(names),
            "genomic_block": {
                "n": len(artifacts.gene_order),
                "columns": f"0..{len(artifacts.gene_order) - 1}",
                "features": list(artifacts.gene_order),
            },
            "clinical_block": {
                "n": len(CLINICAL_FEATURE_NAMES),
                "columns": f"{len(artifacts.gene_order)}..{len(names) - 1}",
                "features": list(CLINICAL_FEATURE_NAMES),
            },
        },
        "data": {
            "background_partition": "train",
            "background_size": len(background_ids),
            "background_patient_ids": list(background_ids),
            "explained_partition": "test",
            "explained_size": len(explained_ids),
            "explained_patient_ids": list(explained_ids),
            "preprocessing": (
                "Train-fold statistics carried inside the checkpoint; nothing "
                "was refitted for this analysis."
            ),
        },
        "explainer": checks["10_environment_recorded"]["explainer"],
        "environment": checks["10_environment_recorded"]["environment"],
        "sanity_checks": checks,
        "tasks": tasks,
        "interpretation": {
            "shap_values_are": (
                "model-attribution values describing how this trained model "
                "distributes its own output across its own inputs"
            ),
            "shap_values_are_not": [
                "causal effects",
                "evidence of a biological mechanism",
                "a validated biomarker ranking",
                "a basis for any clinical decision",
            ],
            "pam50_framing": (
                "The PAM50 task is reproduction/recovery of the established "
                "PAM50 assignment from the same 50-gene panel that assignment "
                "derives from. High attribution to PAM50 genes is therefore "
                "expected by construction and is not independent "
                "molecular-subtype discovery."
            ),
            "additivity": (
                "Expected gradients is a sampling approximation, so the "
                "per-patient SHAP sum only approximately equals "
                "f(x) - E[f(background)]. The recorded additivity gap "
                "quantifies that approximation; it is a diagnostic, not an "
                "error."
            ),
        },
    }


def write_shap_arrays(
    results: dict[str, dict[str, Any]],
    joint_test: torch.Tensor,
    names: Sequence[str],
    modalities: Sequence[str],
    explained_ids: Sequence[str],
    path: Path,
) -> None:
    """Persist the raw SHAP arrays alongside the inputs that produced them.

    Expected gradients is a sampling estimator, so a second run is not
    bit-identical to this one. Saving the arrays means the individual-patient
    explanations, the aggregate rankings and any later re-analysis all quote the
    *same* numbers instead of a fresh estimate that differs in the last digits.

    Keys are ``shap_<task>_<output>`` for each explained output, plus ``features``,
    ``patient_ids``, ``feature_names`` and ``feature_modalities``.
    """
    payload: dict[str, np.ndarray] = {
        "features": joint_test.numpy(),
        "patient_ids": np.array(list(explained_ids)),
        "feature_names": np.array(list(names)),
        "feature_modalities": np.array(list(modalities)),
    }

    for output in EXPLAINED_OUTPUTS:
        result = results[output.task]
        for index, output_name in enumerate(output.output_names):
            key = f"shap_{output.task}_{slug(output_name)}"
            payload[key] = result["arrays"][index]
            payload[f"prediction_{output.task}_{slug(output_name)}"] = result[
                "predictions"
            ][:, index]
            payload[f"base_value_{output.task}_{slug(output_name)}"] = np.asarray(
                result["base_values"][index]
            )

    np.savez_compressed(path, **payload)
    logger.info("Wrote %s (%d arrays).", path.name, len(payload))


def write_outputs(
    results: dict[str, dict[str, Any]],
    joint_test: torch.Tensor,
    names: Sequence[str],
    modalities: Sequence[str],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    explained_ids: Sequence[str],
) -> None:
    """Write every figure, the tables, the raw arrays, and the summary document."""
    ensure_dir(OUTPUT_DIR)
    features = joint_test.numpy()

    for output in EXPLAINED_OUTPUTS:
        result = results[output.task]
        arrays = result["arrays"]
        pooled = np.mean([mean_abs_importance(array) for array in arrays], axis=0)

        plot_global_bar(
            pooled,
            names,
            modalities,
            f"{output.task.upper()} — {output.title}",
            OUTPUT_DIR / f"shap_{output.task}_global.png",
        )

        if len(arrays) == 1:
            plot_beeswarm(
                arrays[0],
                features,
                names,
                f"{output.task.upper()} — {output.title}",
                OUTPUT_DIR / f"shap_{output.task}_beeswarm.png",
            )
        else:
            plot_pam50_class_bars(
                arrays, features, names, OUTPUT_DIR / f"shap_{output.task}_class_bars.png"
            )
            for array, class_name in zip(arrays, output.output_names):
                plot_beeswarm(
                    array,
                    features,
                    names,
                    f"PAM50 {class_name} logit",
                    OUTPUT_DIR / f"shap_{output.task}_beeswarm_{slug(class_name)}.png",
                )

        # Individual-patient figures for the single-output tasks. PAM50's
        # per-class individual attributions are written to the CSV but not
        # plotted, since the subtype target is derived from the same 50-gene
        # panel and the per-class figures would invite over-reading a circular
        # result.
        if len(arrays) != 1:
            continue
        predictions = result["predictions"][:, 0]
        for key, row in representative_indices(predictions).items():
            plot_patient_waterfall(
                arrays[0],
                features,
                names,
                modalities,
                row,
                float(result["base_values"][0]),
                float(predictions[row]),
                f"{output.task.upper()} — {explained_ids[row]} ({key.replace('_', ' ')})",
                OUTPUT_DIR / f"shap_{output.task}_patient_{key}.png",
            )

    write_table(
        OUTPUT_DIR / "shap_feature_importance.csv",
        rows,
        [
            "task",
            "modality",
            "feature",
            "mean_abs_shap",
            "rank",
            "mean_shap",
            "feature_shap_correlation",
            "direction",
        ],
    )
    write_table(
        OUTPUT_DIR / "shap_top_features.csv",
        top_feature_rows(results, names, modalities, features),
        [
            "task",
            "explained_output",
            "rank",
            "modality",
            "feature",
            "mean_abs_shap",
            "mean_shap",
            "feature_shap_correlation",
            "direction",
        ],
    )
    write_table(
        OUTPUT_DIR / "shap_individual_patients.csv",
        individual_rows(results, names, modalities, features, explained_ids),
        [
            "task",
            "explained_output",
            "selection",
            "patient_id",
            "base_value",
            "model_output",
            "sum_all_shap_values",
            "rank",
            "modality",
            "feature",
            "feature_value_standardised",
            "shap_value",
            "pushes_output",
        ],
    )
    write_shap_arrays(
        results,
        joint_test,
        names,
        modalities,
        explained_ids,
        OUTPUT_DIR / "shap_values.npz",
    )

    (OUTPUT_DIR / "shap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def write_table(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    """Write one CSV table with a fixed column order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s.", len(rows), path.name)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> None:
    """Explain the frozen multimodal checkpoint with SHAP."""
    args = parse_args()

    checkpoint_hash_before = sha256_file(CHECKPOINT)
    artifacts = load_model(CHECKPOINT)
    model = artifacts.model
    model.eval()
    parameter_hash_before = sha256_parameters(model)
    checkpoint_epoch = int(
        torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["epoch"]
    )

    config = load_merged_config(TRAIN_CONFIG)
    split = rebuild_split(config)

    metric_check = verify_recorded_metrics(artifacts, split.partitions["test"])
    logger.info(
        "Recorded metrics reproduced: %s (max abs diff %.3e over %d metrics).",
        metric_check["reproduced"],
        metric_check["max_abs_difference"],
        metric_check["metrics_compared"],
    )

    test_patients = list(split.partitions["test"])
    if args.explain_limit is not None:
        test_patients = test_patients[: args.explain_limit]

    background_patients = select_background(
        split.partitions["train"], args.background_size, args.seed
    )

    joint_background, background_ids = build_feature_matrix(
        background_patients, artifacts
    )
    joint_test, explained_ids = build_feature_matrix(test_patients, artifacts)
    names, modalities = feature_names(artifacts)
    n_genes = len(artifacts.gene_order)

    logger.info(
        "Feature space: %d features (%d genomic + %d clinical). "
        "Background %d train patients, explaining %d test patients.",
        len(names),
        n_genes,
        len(CLINICAL_FEATURE_NAMES),
        joint_background.shape[0],
        joint_test.shape[0],
    )

    label_check = verify_label_independence(test_patients, artifacts, joint_test)
    ordering_check = verify_input_ordering(model, joint_test, n_genes)

    results = {
        output.task: explain_output(
            model,
            output,
            joint_background,
            joint_test,
            n_genes,
            args.nsamples,
            args.seed,
        )
        for output in EXPLAINED_OUTPUTS
    }

    checks = run_sanity_checks(
        artifacts,
        results,
        names,
        modalities,
        joint_test,
        checkpoint_hash_before,
        parameter_hash_before,
        metric_check,
        label_check,
        ordering_check,
        args,
    )

    summary = build_summary(
        results,
        names,
        modalities,
        joint_test.numpy(),
        checks,
        artifacts,
        checkpoint_epoch,
        background_ids,
        explained_ids,
        args,
    )
    write_outputs(
        results,
        joint_test,
        names,
        modalities,
        summary,
        importance_rows(results, names, modalities, joint_test.numpy()),
        explained_ids,
    )

    for output in EXPLAINED_OUTPUTS:
        share = summary["tasks"][output.task]["modality_contribution"]["share"]
        top = summary["tasks"][output.task]["outputs"][output.output_names[0]][
            f"top_{TOP_N_TABLE}_features"
        ][:3]
        logger.info(
            "%s: genomic share %.1f%%, clinical share %.1f%%, top features %s.",
            output.task,
            100.0 * share[GENOMIC_MODALITY],
            100.0 * share[CLINICAL_MODALITY],
            [item["feature"] for item in top],
        )


if __name__ == "__main__":
    main()
