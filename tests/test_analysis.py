"""Tests for the Chapter 4 analysis scripts.

These scripts do not train anything, but two of them encode a *scientific*
claim in code, and that is what these tests pin:

- ``run_threshold_analysis`` claims the receptor decision thresholds are chosen
  on validation predictions alone. The selection function must therefore be
  computable from validation rows only, must gate a non-HER2 task on a
  validation-measured gain, and must report threshold-free metrics as
  threshold-free rather than recomputing them at the selected operating point.
- ``export_predictions`` claims both receptor class probabilities are the
  model's own, because reconstructing ``p(negative)`` as ``1 - p(positive)``
  changes the tie structure and perturbs the negative-class average precision.
  The column schema must keep both.
- ``run_shap`` claims a direction of contribution per feature. The direction
  helpers must distinguish a consistent push from a cancelling one, and must
  refuse to name a direction for a feature nobody in the explained cohort
  varies on.

The row dictionaries and arrays below are synthetic fixtures chosen so the
expected arithmetic is checkable by hand. They are not data and no reported
result depends on them.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.export_predictions import field_names
from scripts.run_shap import (
    DIRECTION_CORRELATION_CUTOFF,
    direction_label,
    feature_value_correlation,
    mean_signed_shap,
    representative_indices,
)
from scripts.run_threshold_analysis import (
    DEFAULT_THRESHOLD,
    MIN_VALIDATION_GAIN,
    THRESHOLD_GRID,
    Operating,
    _select,
    apply_to_test,
    operating_point,
    select_thresholds,
    sweep,
)
from src.data.tasks import RECEPTOR_TASKS
from src.evaluation import metrics


def _rows(
    task: str,
    scored: list[tuple[int, float]],
    masked_out: int = 0,
) -> list[dict[str, str]]:
    """Build prediction rows for one task.

    Args:
        task: Receptor task name.
        scored: ``(label, positive-class probability)`` per masked-in patient.
        masked_out: Number of extra patients whose label for ``task`` is absent.

    Returns:
        Rows in the shape ``load_predictions`` returns them, values as strings.
    """
    rows = [
        {f"{task}_mask": "1", f"{task}_true": str(label), f"{task}_prob_pos": str(prob)}
        for label, prob in scored
    ]
    rows.extend(
        {f"{task}_mask": "0", f"{task}_true": "", f"{task}_prob_pos": "0.5"}
        for _ in range(masked_out)
    )
    return rows


#: Separable at the default threshold, so no threshold can improve on 0.5.
ALREADY_OPTIMAL = [(0, 0.10), (0, 0.20), (1, 0.80), (1, 0.90)]

#: Every probability sits below 0.5, so the default threshold predicts nothing
#: positive and a lower threshold recovers perfect separation.
NEEDS_A_LOWER_THRESHOLD = [(0, 0.10), (0, 0.20), (1, 0.30), (1, 0.40)]


# --------------------------------------------------------------------------- #
# Operating-point arithmetic
# --------------------------------------------------------------------------- #
def test_operating_derives_every_rate_from_its_counts() -> None:
    point = Operating(threshold=0.4, tp=6, fp=4, tn=8, fn=2)

    assert point.n == 20
    assert point.sensitivity == pytest.approx(6 / 8)
    assert point.specificity == pytest.approx(8 / 12)
    assert point.precision == pytest.approx(6 / 10)
    assert point.positive_f1 == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))
    assert point.balanced_accuracy == pytest.approx(0.5 * (6 / 8 + 8 / 12))
    assert point.accuracy == pytest.approx(14 / 20)
    assert point.as_dict()["confusion_matrix"] == [[8, 4], [2, 6]]


def test_balanced_accuracy_is_the_mean_of_the_two_recalls() -> None:
    """The reported balanced accuracy must not silently become plain accuracy."""
    imbalanced = Operating(threshold=0.5, tp=1, fp=0, tn=99, fn=9)

    assert imbalanced.accuracy == pytest.approx(100 / 109)
    assert imbalanced.balanced_accuracy == pytest.approx(0.5 * (0.1 + 1.0))


@pytest.mark.parametrize(
    ("point", "expected_zero"),
    [
        (Operating(threshold=0.5, tp=0, fp=0, tn=5, fn=0), "sensitivity"),
        (Operating(threshold=0.5, tp=0, fp=0, tn=0, fn=5), "specificity"),
        (Operating(threshold=0.99, tp=0, fp=0, tn=5, fn=5), "precision"),
    ],
)
def test_degenerate_partitions_return_zero_rather_than_dividing_by_zero(
    point: Operating, expected_zero: str
) -> None:
    """A collapsed model must produce a number, not an exception."""
    assert getattr(point, expected_zero) == 0.0
    assert point.positive_f1 == 0.0


# --------------------------------------------------------------------------- #
# Thresholding semantics
# --------------------------------------------------------------------------- #
def test_a_probability_equal_to_the_threshold_counts_as_positive() -> None:
    """The rule is ``prob >= threshold``; the boundary must not drift."""
    y_true = np.array([1, 1])
    prob_pos = np.array([0.30, 0.29])

    point = operating_point(y_true, prob_pos, 0.30)

    assert (point.tp, point.fn) == (1, 1)


def test_candidate_grid_is_the_declared_interior_grid() -> None:
    """0.0 and 1.0 are excluded: both make one class unreachable."""
    assert len(THRESHOLD_GRID) == 99
    assert THRESHOLD_GRID[0] == 0.01
    assert THRESHOLD_GRID[-1] == 0.99
    assert DEFAULT_THRESHOLD in THRESHOLD_GRID
    assert 0.0 not in THRESHOLD_GRID and 1.0 not in THRESHOLD_GRID


def test_sweep_covers_the_whole_grid_in_order() -> None:
    y_true, prob_pos = np.array([0, 1]), np.array([0.2, 0.8])

    points = sweep(y_true, prob_pos)

    assert [point.threshold for point in points] == list(THRESHOLD_GRID)


def test_selection_breaks_ties_toward_the_default_threshold() -> None:
    """Among equally good thresholds, prefer the least surprising one."""
    tied = [Operating(threshold=t, tp=1, fp=1, tn=1, fn=1) for t in (0.30, 0.45, 0.70)]

    assert _select(tied, "balanced_accuracy").threshold == 0.45


def test_equidistant_ties_break_toward_the_smaller_threshold() -> None:
    """A documented, deterministic rule, so the choice is reproducible."""
    tied = [Operating(threshold=t, tp=1, fp=1, tn=1, fn=1) for t in (0.40, 0.60)]

    assert _select(tied, "balanced_accuracy").threshold == 0.40


# --------------------------------------------------------------------------- #
# Validation-only selection
# --------------------------------------------------------------------------- #
def test_selection_ignores_patients_whose_label_is_absent() -> None:
    """Masked-out patients must not enter the threshold sweep."""
    selection = select_thresholds(_rows("er", ALREADY_OPTIMAL, masked_out=7), "er")

    assert selection["n_validation_scored"] == len(ALREADY_OPTIMAL)
    assert selection["n_validation_positive"] == 2
    assert selection["n_validation_negative"] == 2


def test_selection_reads_only_validation_rows() -> None:
    """The selection payload must be computable without any test row.

    Passing validation rows alone is enough to produce a complete selection,
    including the decision on whether to touch the test partition. That is the
    structural guarantee behind the validation-only claim.
    """
    selection = select_thresholds(_rows("pr", NEEDS_A_LOWER_THRESHOLD), "pr")

    assert selection["partition_used_for_selection"] == "val"
    assert not any("test" in key for key in selection if key != "applied_to_test")
    assert selection["primary_threshold"] in THRESHOLD_GRID
    assert selection["secondary_threshold"] in THRESHOLD_GRID


def test_a_task_already_optimal_at_the_default_threshold_is_not_applied() -> None:
    """No validation gain means the test partition is left alone."""
    selection = select_thresholds(_rows("er", ALREADY_OPTIMAL), "er")

    assert selection["primary_threshold"] == DEFAULT_THRESHOLD
    assert selection["validation_balanced_accuracy_gain"] == pytest.approx(0.0)
    assert selection["applied_to_test"] is False


def test_a_task_with_a_real_validation_gain_is_applied() -> None:
    selection = select_thresholds(_rows("pr", NEEDS_A_LOWER_THRESHOLD), "pr")

    assert selection["validation_at_default_threshold"]["balanced_accuracy"] == 0.5
    assert selection["validation_at_primary_threshold"]["balanced_accuracy"] == 1.0
    assert selection["validation_balanced_accuracy_gain"] >= MIN_VALIDATION_GAIN
    assert selection["applied_to_test"] is True


def test_her2_is_applied_even_without_a_validation_gain() -> None:
    """HER2 motivated the analysis, so it is reported either way."""
    selection = select_thresholds(_rows("her2", ALREADY_OPTIMAL), "her2")

    assert selection["validation_balanced_accuracy_gain"] == pytest.approx(0.0)
    assert selection["applied_to_test"] is True
    assert "unconditionally" in selection["application_reason"]


# --------------------------------------------------------------------------- #
# Applying the selected threshold to test
# --------------------------------------------------------------------------- #
def test_threshold_free_metrics_are_reported_threshold_free() -> None:
    """Moving the threshold walks the same curve; the curve must not be rescored."""
    validation = _rows("her2", NEEDS_A_LOWER_THRESHOLD)
    test = _rows("her2", [(0, 0.15), (1, 0.35), (0, 0.45), (1, 0.25)])
    selection = select_thresholds(validation, "her2")

    applied = apply_to_test(test, "her2", selection)

    y_true = np.array([0, 1, 0, 1])
    prob_pos = np.array([0.15, 0.35, 0.45, 0.25])
    unchanged = applied["threshold_free_metrics_unchanged"]
    assert unchanged["roc_auc"] == pytest.approx(metrics.roc_auc(y_true, prob_pos))
    assert unchanged["average_precision_positive_class"] == pytest.approx(
        metrics.pr_auc(y_true, prob_pos)
    )


def test_applying_a_threshold_reports_the_change_against_the_default() -> None:
    """The default operating point must stay in the payload for comparison."""
    validation = _rows("her2", NEEDS_A_LOWER_THRESHOLD)
    test = _rows("her2", [(0, 0.10), (1, 0.30), (1, 0.40), (0, 0.20)])
    selection = select_thresholds(validation, "her2")

    applied = apply_to_test(test, "her2", selection)

    assert applied["test_at_default_threshold"]["sensitivity"] == 0.0
    assert applied["test_at_primary_threshold"]["sensitivity"] == 1.0
    assert applied["primary_change_vs_default"]["sensitivity"] == pytest.approx(1.0)
    assert applied["n_test_positive"] == 2


# --------------------------------------------------------------------------- #
# Exported prediction schema
# --------------------------------------------------------------------------- #
def test_both_receptor_class_probabilities_are_exported() -> None:
    """The negative column is stored, not reconstructed from the positive one."""
    columns = field_names()

    for task in RECEPTOR_TASKS:
        assert f"{task}_prob_neg" in columns
        assert f"{task}_prob_pos" in columns
    assert len(columns) == len(set(columns))


# --------------------------------------------------------------------------- #
# SHAP direction of contribution
# --------------------------------------------------------------------------- #
def test_mean_signed_shap_separates_a_consistent_push_from_a_cancelling_one() -> None:
    """A large magnitude with a near-zero signed mean is the cancelling case."""
    array = np.array([[0.4, 0.4], [0.4, -0.4], [0.4, 0.4], [0.4, -0.4]])

    signed = mean_signed_shap(array)

    assert signed[0] == pytest.approx(0.4)
    assert signed[1] == pytest.approx(0.0)
    assert np.abs(array).mean(axis=0)[1] == pytest.approx(0.4)


def test_feature_value_correlation_recovers_a_monotone_direction() -> None:
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    array = np.array([[-0.2], [-0.1], [0.1], [0.2]])

    assert feature_value_correlation(array, features)[0] == pytest.approx(1.0)
    assert feature_value_correlation(-array, features)[0] == pytest.approx(-1.0)


def test_a_feature_nobody_varies_on_has_no_defined_direction() -> None:
    """A one-hot level absent from the explained cohort must not get a direction."""
    features = np.array([[0.0], [0.0], [0.0]])
    array = np.array([[0.1], [-0.2], [0.3]])

    correlation = feature_value_correlation(array, features)[0]

    assert np.isnan(correlation)
    assert direction_label(correlation).startswith("undefined")


@pytest.mark.parametrize(
    ("correlation", "expected"),
    [
        (0.9, "higher feature value increases the output"),
        (-0.9, "higher feature value decreases the output"),
        (0.0, "mixed (no consistent direction across patients)"),
    ],
)
def test_direction_label_names_the_three_resolvable_cases(
    correlation: float, expected: str
) -> None:
    assert direction_label(correlation) == expected


def test_direction_label_is_undecided_inside_the_cutoff() -> None:
    """Weak correlations must be reported as mixed, not given a false direction."""
    inside = DIRECTION_CORRELATION_CUTOFF / 2

    assert direction_label(inside).startswith("mixed")
    assert direction_label(-inside).startswith("mixed")
    assert direction_label(DIRECTION_CORRELATION_CUTOFF).startswith("higher")


# --------------------------------------------------------------------------- #
# Representative patient selection
# --------------------------------------------------------------------------- #
def test_representative_patients_are_chosen_by_model_output_alone() -> None:
    predictions = np.array([0.5, -2.0, 3.0, 1.0, 0.0])

    chosen = representative_indices(predictions)

    assert chosen["highest_output"] == 2
    assert chosen["lowest_output"] == 1
    assert predictions[chosen["median_output"]] == pytest.approx(0.5)


def test_the_median_representative_is_a_real_patient() -> None:
    """An interpolated median would describe a patient who does not exist."""
    predictions = np.array([0.0, 1.0, 2.0, 3.0])

    chosen = representative_indices(predictions)

    assert chosen["median_output"] in range(len(predictions))
    assert predictions[chosen["median_output"]] in set(predictions.tolist())
