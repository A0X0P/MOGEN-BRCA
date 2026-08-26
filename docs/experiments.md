# Experiments

This document records the experiments that have actually been run, the commands
that reproduce them, and the caveats that must accompany the numbers. Every
figure quoted here is read back from a committed artefact under `results/`; none
is copied from a paper or estimated.

Three runs exist. All three share one patient-level split, one seed, and one
optimisation protocol, so a difference between them is attributable to the
modality configuration rather than to retuning.

| Run | Directory | Input | Status |
| --- | --- | --- | --- |
| Full multimodal | `results/breast/` | 50 PAM50 genes + 12-dim clinical | frozen reference |
| Genomics-only ablation | `results/breast_ablation/genomics_only/` | 50 PAM50 genes | ablation |
| Clinical-only ablation | `results/breast_ablation/clinical_only/` | 12-dim clinical | ablation |

`results/breast/` is the frozen reference run. It is never rewritten: the
ablation configs write only under `results/breast_ablation/`, and
`tests/test_ablations.py` asserts that property directly against the checked-in
YAML so it cannot regress silently.

The post-hoc analyses — detailed metrics, threshold analysis, SHAP attribution and
the literature record — read the frozen checkpoint and write only into
`results/breast_analysis/` and `results/breast_explainability/`. Nothing is
written inside `results/breast/` at any point, and the checkpoint's SHA-256 is
recorded before and after the SHAP run to prove it
(`1ad26a2535f1479e621429f73306f3be4d84e1ce878e5fde28f283a82b5faf14`, unchanged).

---

## 1. Reproducing the runs

```bash
uv run pytest                                     # 303 tests

# Ablation training (each writes only into its own results directory)
uv run python scripts/run_train.py --config configs/breast/train_genomics_only.yaml
uv run python scripts/run_train.py --config configs/breast/train_clinical_only.yaml

# Independent test-partition evaluation of each best checkpoint
uv run python scripts/run_eval.py \
    --config configs/breast/train_genomics_only.yaml \
    --checkpoint results/breast_ablation/genomics_only/checkpoints/checkpoint_best.pt \
    --partition test

uv run python scripts/run_eval.py \
    --config configs/breast/train_clinical_only.yaml \
    --checkpoint results/breast_ablation/clinical_only/checkpoints/checkpoint_best.pt \
    --partition test

# Cross-run comparison (read-only with respect to every run it compares)
uv run python scripts/run_ablation_comparison.py

# SHAP attribution for the frozen full multimodal checkpoint
uv run python scripts/run_shap.py

# Per-patient prediction export, per run and per partition. --output is explicit
# so nothing can be written into a frozen run's directory.
uv run python scripts/export_predictions.py \
    --config configs/breast/train.yaml \
    --checkpoint results/breast/checkpoints/checkpoint_best.pt \
    --partition test \
    --output results/breast_analysis/predictions/full_multimodal_test.csv

# Confusion matrices, positive-class detail, and the PR-AUC disambiguation.
# Cross-checks itself against every committed test_metrics.json.
uv run python scripts/run_detailed_metrics.py

# Receptor threshold selection on validation, applied once to test
uv run python scripts/run_threshold_analysis.py
```

`scripts/run_ablation_comparison.py` and `scripts/run_shap.py` never write into
`results/breast/`; the SHAP script additionally re-hashes the frozen checkpoint
before and after the analysis and fails if the two hashes differ.

`scripts/run_detailed_metrics.py` and `scripts/run_threshold_analysis.py` read
the exported predictions and write only into `results/breast_analysis/`. The
detailed-metrics script recomputes every quantity that also appears in a
committed `test_metrics.json` and aborts if any of them disagrees beyond `1e-6`,
so a drift between the export path and the evaluator cannot pass unnoticed.

---

## 2. How the ablations are constructed

An ablation is a *configuration* of the ratified architecture, not a second
architecture. `configs/breast/model_genomics_only.yaml` and
`configs/breast/model_clinical_only.yaml` each differ from
`configs/breast/model.yaml` in exactly one key — `enable_clinical: false` and
`enable_genomics: false` respectively. Every width, depth, dropout rate and head
dimension is unchanged, and a test asserts that the symmetric difference of the
two config dictionaries is exactly that one key.

When a modality is disabled, `build_model` omits its encoder and omits the
cross-modal attention module entirely rather than instantiating it and leaving it
inert. Passing a tensor for a disabled modality raises `ValueError` instead of
being silently ignored.

The fusion head input width is fixed at `embedding_dim * 2 + 2`. A missing
modality contributes a zero block plus a zero presence indicator, so the fusion
stage and all five heads keep their exact shapes across the three runs. This is
why parameter counts differ only by the removed encoder:

| Run | Trainable parameters |
| --- | --- |
| Full multimodal | 1,514,512 |
| Genomics-only | 885,392 |
| Clinical-only | 483,728 |

All five tasks are retained in every run — PAM50 5-class, ER, PR, HER2, and
DeepSurv survival — with per-task masking unchanged. No task is dropped to suit
an ablation.

### Protocol identity

Both ablation training configs copy the frozen run's optimisation protocol
verbatim: seed 42, 60-epoch budget, batch size 32, learning rate 3e-4, weight
decay 1e-5, plateau scheduler, `monitor: val_total` (min), early-stopping
patience 12, `mixed_precision: false`, and the same `configs/breast/data.yaml`.
`tests/test_ablations.py` pins each of those keys against `train.yaml`.

Model selection used the validation partition only. Preprocessing statistics
(gene-wise standardisation and clinical normalisation) are fitted on the training
fold alone and carried inside each checkpoint, so evaluation re-uses them rather
than refitting.

### Split integrity

The comparison script re-reads all three `split.json` files and refuses to emit a
comparison unless every partition matches the frozen reference in **both
membership and order**, and unless all pairwise train/val/test overlaps are
empty. That check passes: train 756, validation 163, test 163.

---

## 3. Ablation training outcomes

Duration is the wall-clock span between the first and last timestamped line of
the committed `train_stdout.log`, on CPU (`cuda_available: false`). It excludes
interpreter start-up, so it slightly understates total process time.
`run_summary.json` does not record a duration field; the log is the audit trail.

Best epoch is the value stored in the checkpoint, which is **0-based** — epoch 14
is the fifteenth epoch. This matches the frozen run's recorded best epoch of 14.

| Run | Best epoch | Best `val_total` | Epochs completed | Early stopping | Duration |
| --- | --- | --- | --- | --- | --- |
| Full multimodal | 14 | 1.680673 | 27 / 60 | triggered | not logged |
| Genomics-only | 28 | 1.592172 | 41 / 60 | triggered | 361 s (6 m 01 s) |
| Clinical-only | 14 | 2.435860 | 27 / 60 | triggered | 77 s (1 m 17 s) |

All three runs early-stopped before exhausting the 60-epoch budget, each after 12
consecutive epochs without a `val_total` improvement (patience 12). The
genomics-only log ends with `Training stopped at epoch 40`; the clinical-only log
ends with `Early stopping triggered at epoch 26`.

Note that the three runs selected **different** epochs under the identical
protocol. This is a genuine confound and is recorded in the comparison file's
`caveats`: `val_total` is dominated by the Cox term, so the selected epoch is
driven largely by survival behaviour, and the reported per-task differences
conflate the modality effect with the selection epoch.

---

## 4. Test-partition results

Evaluated once per run on the held-out test partition, using each run's own best
checkpoint. Per-task `n` differs because labels are masked per task, never
imputed and never filled by a default class.

**Balanced accuracy is macro recall.** For a single-label task the mean of the
per-class recalls *is* balanced accuracy, so the comparison maps the evaluator's
existing `recall` field onto the `balanced_accuracy` column rather than adding a
second implementation of the same quantity.

**The `PR-AUC` column for ER, PR and HER2 is a macro over both classes, not the
conventional binary average precision.** `src/evaluation/evaluator.py` computes
`softmax(logits)` and passes the full `(N, 2)` matrix to `metrics.pr_auc`, which
macro-averages the per-class average precision. So every `pr_auc` in a committed
`test_metrics.json` — and every receptor PR-AUC in the tables below — is
`mean(AP_positive, AP_negative)`. ROC-AUC and Brier are conventional: they slice
the positive-class column. This was found while cross-checking the export path
against the evaluator and is a *definition* difference, not an error, but it
matters because averaging in the easy negative class flatters an imbalanced task.
On HER2 the difference is large: 0.6963 macro versus **0.4758** positive-class
average precision for the full model. `scripts/run_detailed_metrics.py` reports
both under `pr_auc_macro_both_classes` and `pr_auc_positive_class`. The frozen
files were not rewritten.

Majority-class accuracy on this test partition, as a descriptive floor:
subtype 0.5135, ER 0.7871, PR 0.6645, HER2 0.7612.

### Full multimodal (frozen reference)

| Task | n | Accuracy | Balanced acc. | Macro-F1 | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PAM50 | 148 | 0.8649 | 0.7609 | 0.7904 | 0.9430 | 0.8045 | — |
| ER | 155 | 0.9097 | 0.8874 | 0.8708 | 0.9232 | 0.8823 | 0.0827 |
| PR | 155 | 0.8645 | 0.8124 | 0.8345 | 0.8663 | 0.8715 | 0.1427 |
| HER2 | 134 | 0.7761 | 0.5741 | 0.5765 | 0.7736 | 0.6963 | 0.1672 |
| Survival | 160 | C-index 0.7112 (23 events) | | | | | |

### Genomics-only

| Task | n | Accuracy | Balanced acc. | Macro-F1 | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PAM50 | 148 | 0.8649 | 0.7721 | 0.7912 | 0.9622 | 0.8432 | — |
| ER | 155 | 0.9226 | 0.8845 | 0.8845 | 0.9183 | 0.8763 | 0.0822 |
| PR | 155 | 0.8581 | 0.7980 | 0.8230 | 0.8652 | 0.8763 | 0.1473 |
| HER2 | 134 | 0.8284 | 0.6728 | 0.7033 | 0.8033 | 0.7930 | 0.1437 |
| Survival | 160 | C-index 0.6008 (23 events) | | | | | |

### Clinical-only

| Task | n | Accuracy | Balanced acc. | Macro-F1 | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PAM50 | 148 | 0.4932 | 0.2082 | 0.1649 | 0.4505 | 0.2174 | — |
| ER | 155 | 0.7871 | 0.5000 | 0.4404 | 0.4773 | 0.5022 | 0.2008 |
| PR | 155 | 0.6645 | 0.5000 | 0.3992 | 0.4979 | 0.5318 | 0.2362 |
| HER2 | 134 | 0.7612 | 0.5000 | 0.4322 | 0.5411 | 0.5454 | 0.1883 |
| Survival | 160 | C-index 0.7372 (23 events) | | | | | |

On all three receptor tasks the clinical-only model's accuracy equals the
majority-class rate and its balanced accuracy is exactly 0.5000, with ROC-AUC at
or below chance. That is majority-class collapse: the model predicts one class
for every patient. On PAM50 its accuracy (0.4932) is *below* the majority-class
rate (0.5135). This is the expected outcome for a deliberately untuned
single-modality baseline and is reported as measured — the baseline was not
redesigned to look competitive.

---

## 5. Differences

`results/breast_ablation/ablation_comparison.{json,csv}` holds these at full
precision. Differences are plain arithmetic on the two test scores.

**Sign convention:** for accuracy, balanced accuracy, macro-F1, ROC-AUC, PR-AUC
and C-index, higher is better, so a positive difference favours the full model.
Brier score is lower-is-better, so a **positive** Brier difference means the full
model is *worse* calibrated on that task.

### Full − genomics-only

| Task | Accuracy | Balanced acc. | Macro-F1 | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| PAM50 | 0.0000 | −0.0112 | −0.0008 | −0.0192 | −0.0387 | — |
| ER | −0.0129 | +0.0029 | −0.0137 | +0.0050 | +0.0060 | +0.0005 |
| PR | +0.0065 | +0.0144 | +0.0115 | +0.0011 | −0.0048 | −0.0046 |
| HER2 | −0.0522 | −0.0987 | −0.1268 | −0.0297 | −0.0968 | +0.0235 |
| Survival | C-index +0.1104 | | | | | |

Adding the clinical modality does **not** uniformly help. It gains substantially
on survival (+0.1104 C-index) and is roughly neutral on PAM50, ER and PR, but it
*loses* on HER2 across every classification metric. On this evidence the benefit
of fusion in this implementation is concentrated in the survival task.

### Full − clinical-only

| Task | Accuracy | Balanced acc. | Macro-F1 | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| PAM50 | +0.3716 | +0.5527 | +0.6255 | +0.4925 | +0.5872 | — |
| ER | +0.1226 | +0.3874 | +0.4304 | +0.4460 | +0.3801 | −0.1181 |
| PR | +0.2000 | +0.3124 | +0.4353 | +0.3685 | +0.3397 | −0.0935 |
| HER2 | +0.0149 | +0.0741 | +0.1443 | +0.2325 | +0.1509 | −0.0210 |
| Survival | C-index −0.0259 | | | | | |

Gene expression carries essentially all of the receptor and subtype signal.
Survival is the sole task where the clinical-only model scores *higher* than the
full model.

### No significance testing was performed

`ablation_comparison.json` records `statistical_testing.performed: false`. No
hypothesis test, confidence interval, or resampling procedure was run, so none of
these differences may be described as statistically significant. With 148–160
scored test patients per task and only 23 survival events, differences of a few
points are well within the range that sampling variation alone could produce.

---

## 6. Confusion matrices and positive-class detail

`scripts/run_detailed_metrics.py` writes
`results/breast_analysis/detailed_metrics.{json,csv}` from the exported
per-patient predictions. It recomputes 29 quantities per run that also appear in
that run's committed `test_metrics.json` and aborts if any disagree beyond
`1e-6`. All three runs agree: maximum absolute differences of 9.069e-09
(full multimodal), 1.351e-08 (genomics-only) and 6.791e-09 (clinical-only).

Every confusion matrix below is oriented **rows = true class, columns =
predicted class**.

### PAM50, 5-class

Class order is Luminal A, Luminal B, HER2-enriched, Basal-like, Normal-like, with
test support 76 / 30 / 12 / 25 / 5.

| Run | Macro precision | Balanced acc. | Confusion matrix |
| --- | --- | --- | --- |
| Full multimodal | 0.8714 | 0.7609 | `[[70,5,1,0,0],[7,22,1,0,0],[0,3,9,0,0],[0,0,0,25,0],[1,1,1,0,2]]` |
| Genomics-only | 0.8301 | 0.7721 | `[[73,0,0,0,3],[11,19,0,0,0],[0,4,8,0,0],[0,0,0,25,0],[1,1,0,0,3]]` |
| Clinical-only | 0.1506 | 0.2082 | `[[70,0,0,6,0],[27,0,0,3,0],[11,1,0,0,0],[22,0,0,3,0],[4,0,0,1,0]]` |

Both genomic-bearing runs classify Basal-like perfectly (25/25) and fail mainly
on Normal-like, the smallest class at 5 patients. The full model's errors
concentrate on the Luminal A / Luminal B boundary, which is the biologically
least separable pair in PAM50. The clinical-only matrix has empty columns: over
148 patients it predicts Luminal A 134 times, Basal-like 13 times, Luminal B
once, and **HER2-enriched and Normal-like never**.

### Receptor tasks

`AP+` is the conventional positive-class average precision; `AP macro` is the
macro-over-both-classes figure that the committed files call `pr_auc`.

| Run | Task | n | pos | Acc. | Bal. acc. | ROC-AUC | AP+ | AP macro | Brier | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full | ER | 155 | 122 | 0.9097 | 0.8874 | 0.9232 | 0.9670 | 0.8823 | 0.0827 | `[[28,5],[9,113]]` |
| Full | PR | 155 | 103 | 0.8645 | 0.8124 | 0.8663 | 0.9104 | 0.8715 | 0.1427 | `[[34,18],[3,100]]` |
| Full | HER2 | 134 | 32 | 0.7761 | 0.5741 | 0.7736 | 0.4758 | 0.6963 | 0.1672 | `[[98,4],[26,6]]` |
| Genomics | ER | 155 | 122 | 0.9226 | 0.8845 | 0.9183 | 0.9521 | 0.8763 | 0.0822 | `[[27,6],[6,116]]` |
| Genomics | PR | 155 | 103 | 0.8581 | 0.7980 | 0.8652 | 0.9097 | 0.8763 | 0.1473 | `[[32,20],[2,101]]` |
| Genomics | HER2 | 134 | 32 | 0.8284 | 0.6728 | 0.8033 | 0.6887 | 0.7930 | 0.1437 | `[[99,3],[20,12]]` |
| Clinical | ER | 155 | 122 | 0.7871 | 0.5000 | 0.4773 | 0.7628 | 0.5022 | 0.2008 | `[[0,33],[0,122]]` |
| Clinical | PR | 155 | 103 | 0.6645 | 0.5000 | 0.4979 | 0.6479 | 0.5318 | 0.2362 | `[[0,52],[0,103]]` |
| Clinical | HER2 | 134 | 32 | 0.7612 | 0.5000 | 0.5411 | 0.2901 | 0.5454 | 0.1883 | `[[102,0],[32,0]]` |

### HER2 positive-class metrics

HER2 is the weak, imbalanced task (32 positives of 134), so it is reported at the
positive-class level rather than only as a macro average.

| Run | Precision | Recall | F1 | Specificity | Predicted positive | TP of 32 |
| --- | --- | --- | --- | --- | --- | --- |
| Full multimodal | 0.6000 | 0.1875 | 0.2857 | 0.9608 | 10 | 6 |
| Genomics-only | 0.8000 | 0.3750 | 0.5106 | 0.9706 | 15 | 12 |
| Clinical-only | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |

The full multimodal model finds **6 of 32** HER2-positive patients at the default
threshold. Its accuracy of 0.7761 is almost entirely specificity: it is barely
above the 0.7612 majority-class floor. Genomics-only is better on every one of
these columns.

For ER and PR the positive-class figures are strong for both genomic runs — full
model ER precision 0.9576 / recall 0.9262 / F1 0.9417, PR 0.8475 / 0.9709 /
0.9050; genomics-only ER 0.9508 / 0.9508 / 0.9508, PR 0.8347 / 0.9806 / 0.9018.

### Majority-class collapse is directional

The clinical-only run does not merely score badly; its confusion matrices have an
all-zero column on every receptor task. It predicts **positive for all 155**
patients on ER and PR, and **negative for all 134** on HER2. The collapse follows
the majority class of each task, and its ROC-AUC is at or below chance on ER
(0.4773) and PR (0.4979), so there is no usable ranking underneath the collapsed
decision either. This is reported as measured; the baseline was not retuned.

### Survival, all three runs

Identical evaluable sets, because survival eligibility depends on the label rules
rather than on the model: **n = 160, 23 events, 137 censored**, median follow-up
28.52 months. C-index 0.7112 (full), 0.6008 (genomics-only), 0.7372
(clinical-only).

---

## 7. Receptor decision thresholds, selected on validation

`scripts/run_threshold_analysis.py` writes
`results/breast_analysis/threshold_analysis.json` and `threshold_sweep.csv`
(891 rows). It was motivated by the full model detecting only 6 of 32
HER2-positive test patients at a 0.5 threshold.

### Protocol

The rule was fixed before any test label was read:

1. Sweep thresholds 0.01–0.99 in 0.01 steps over the **validation** partition.
2. **Primary rule:** maximise validation balanced accuracy. Ties break toward the
   threshold nearest 0.50, then toward the smaller value.
3. **Secondary rule:** the same, maximising validation positive-class F1. Both are
   selected before test is touched and both are applied once, so neither is a
   post-hoc pick between two test outcomes.
4. A selected threshold is applied to test if the task is HER2, or if its
   validation balanced-accuracy gain over 0.50 is at least 0.02.

`main()` enforces the ordering structurally: every threshold is selected and
written into the payload in phase 1, and test predictions are not loaded until
phase 2. `tests/test_analysis.py` pins the gating, the tie-breaking, and the
`prob >= threshold` boundary.

**Thresholding cannot improve ROC-AUC or average precision.** Those rank
patients and are identical at every threshold. Anything below is a movement along
the same curve, not a better model.

### Full multimodal

| Task | Selected | Val. bal. acc. | Applied? | Test bal. acc. | Test sens. | Test pos. F1 |
| --- | --- | --- | --- | --- | --- | --- |
| ER | 0.50 | 0.9279 → 0.9279 (+0.0000) | no | — | — | — |
| PR | 0.63 | 0.8122 → 0.8437 (+0.0315) | yes | 0.8124 → **0.7926** | 0.9709 → 0.8544 | 0.9050 → 0.8585 |
| HER2 | 0.36 | 0.7028 → 0.7615 (+0.0588) | yes | 0.5741 → **0.6198** | 0.1875 → 0.4062 | 0.2857 → 0.4194 |

**HER2 is the intended result.** Lowering the threshold to 0.36 more than doubles
detected positives, from 6 to **13 of 32**, and raises balanced accuracy from
0.5741 to 0.6198. It is a real trade: accuracy falls 0.7761 → 0.7313, positive
precision falls 0.6000 → 0.4333, and false positives rise from 4 to 17. Which
operating point is preferable is a clinical judgement, not a modelling one.

**PR is a negative result and is reported as one.** The validation rule selected
0.63, but on test it made balanced accuracy *worse*, 0.8124 → 0.7926. The
pre-registered secondary threshold of 0.61 gave 0.8217. A threshold chosen on 163
validation patients did not transfer. This is exactly the failure mode that
selecting on test would have hidden.

**ER was correctly gated out.** Its validation optimum already was 0.50, gain
0.0000, so the test partition was not re-scored for ER at all.

### Genomics-only

| Task | Selected | Val. bal. acc. | Test bal. acc. | Test sens. | Test acc. |
| --- | --- | --- | --- | --- | --- |
| ER | 0.58 | 0.9065 → 0.9319 | 0.8845 → 0.8997 | 0.9508 → 0.9508 | 0.9226 → 0.9290 |
| PR | 0.61 | 0.8069 → 0.8322 | 0.7980 → 0.8024 | 0.9806 → 0.8932 | 0.8581 → 0.8323 |
| HER2 | 0.37 | 0.7515 → 0.7887 | 0.6728 → **0.7264** | 0.3750 → 0.5312 | 0.8284 → 0.8284 |

Genomics-only at 0.37 is the **best HER2 operating point measured anywhere in
this study**: 17 of 32 positives detected, balanced accuracy 0.7264, positive F1
0.5106 → 0.5965, and accuracy unchanged at 0.8284. Unlike the full model's HER2
retune, this one costs nothing in overall accuracy.

### Clinical-only

Every selected threshold produced test balanced accuracy at or below chance —
ER 0.5060, PR 0.4791, HER2 0.4804 — despite apparent validation gains of +0.10,
+0.09 and +0.09. This is the expected consequence of at-or-below-chance ROC-AUC:
moving a threshold redistributes errors but cannot manufacture ranking signal
that the model does not have.

---

## 8. SHAP attribution for the frozen model

`scripts/run_shap.py` explains the frozen full multimodal checkpoint. It never
retrains, never steps an optimiser, and never uses test labels.

### What is explained

The model takes two tensors, which SHAP cannot address directly, so the script
wraps it in `JointInputModel`: a single 62-column input where columns 0–49 are
the PAM50 genes **in the checkpoint's own recorded `gene_order`** and columns
50–61 are the clinical vector in `CLINICAL_FEATURE_NAMES` order
(`age`, `stage_I…stage_IV`, `stage_Unknown`, `nodal_N0…nodal_N3`,
`nodal_Unknown`, `sex_male`). The wrapper splits the matrix and calls the model's
own `forward`; a test asserts the wrapper's output is bit-identical
(`torch.equal`) to the direct two-tensor call. No surrogate or re-implementation
is involved.

Explainer: `shap.GradientExplainer` (expected gradients) — PyTorch-native and
able to handle the LayerNorm, MultiheadAttention and GELU operations in the
genomic transformer. Background: 100 **training** patients (seed 42). Explained:
all 163 **test** patients. `nsamples: 200` per patient.

Explained outputs:

- **ER / PR / HER2** — the logit margin `logits[:, 1] − logits[:, 0]`, which is
  monotone in P(positive), so attribution sign reads directly as "pushes toward
  positive/negative".
- **PAM50** — all five class logits, giving class-specific attributions for
  Luminal A, Luminal B, HER2-enriched, Basal-like and Normal-like.
- **Survival** — the raw DeepSurv risk score.

Because expected gradients is a sampling approximation, the script records an
additivity diagnostic per task (mean and max |Σ SHAP − (prediction − base
value)|) in `shap_summary.json` rather than asserting exact additivity. The
observed gaps are roughly 8–25 % of the standard deviation of the deviation being
attributed; the rankings below should be read as approximate.

### Modality split of attribution

Share of total mean |SHAP| by modality:

| Task | Genomic | Clinical |
| --- | --- | --- |
| ER | 87.9 % | 12.1 % |
| PR | 91.1 % | 8.9 % |
| HER2 | 87.6 % | 12.4 % |
| PAM50 | 88.3 % | 11.7 % |
| **Survival** | **42.2 %** | **57.8 %** |

Survival is the one task where the clinical block carries the majority of
attribution — which agrees independently with the ablation result that the
clinical-only model attains the best C-index.

### Top-ranked features

| Task | Highest mean \|SHAP\| features |
| --- | --- |
| ER | ESR1 0.2719, FOXC1 0.1547, MAPT 0.0638, NAT1 0.0531, ANLN 0.0459, ERBB2 0.0399 |
| PR | ESR1 0.1716, FOXC1 0.1350, MAPT 0.0468, CEP55 0.0394, NAT1 0.0370, ANLN 0.0312 |
| HER2 | FOXC1 0.1451, ANLN 0.0577, ESR1 0.0565, MAPT 0.0397, CEP55 0.0390, CDC6 0.0344 |
| PAM50 (mean over classes) | ESR1 0.3775, FOXC1 0.2913, CEP55 0.0955, ANLN 0.0893, MAPT 0.0846, NAT1 0.0678 |
| Luminal A | ESR1 0.4843, FOXC1 0.3191, MAPT 0.1420, ANLN 0.1277, NAT1 0.1147 |
| Luminal B | ESR1 0.4676, FOXC1 0.3869, CEP55 0.0946, KRT17 0.0755, CDH3 0.0731 |
| HER2-enriched | ESR1 0.3270, ANLN 0.1746, MAPT 0.1564, CEP55 0.1257, CDC6 0.1218 |
| Basal-like | FOXC1 0.5590, ESR1 0.5095, CDH3 0.0957, CEP55 0.0788, CCNE1 0.0651 |
| Normal-like | ESR1 0.0989, FOXC1 0.0870, CEP55 0.0678, age 0.0505, KRT17 0.0473 |
| **Survival** | **age 0.4652, nodal_N0 0.3610, nodal_N2 0.1345, stage_III 0.1304, stage_II 0.1204**, MAPT 0.0830 |

Two observations worth stating plainly, because neither flatters the model:

- **ERBB2 does not rank in the top six for the HER2 task.** The HER2 head's
  attribution is instead led by FOXC1 and proliferation-associated genes. Given
  that HER2 is also the model's weakest task (macro-F1 0.5765, and the one task
  where the genomics-only ablation beats the full model), the most defensible
  reading is that the HER2 head has not learned an ERBB2-centred rule.
- The survival explanation is led entirely by clinical variables — age, nodal
  stage, tumour stage — with the first gene appearing sixth.

Full rankings for all 62 features across all 10 explained outputs (5 tasks, with
PAM50 additionally broken out per class) are in
`results/breast_explainability/shap_feature_importance.csv` (620 rows), and the
top 20 per output in `shap_top_features.csv` (180 rows).

### Direction of contribution

Magnitude alone does not say which way a feature pushes. Each row of
`shap_feature_importance.csv` therefore also carries `mean_signed_shap`, the
Pearson correlation between the feature's standardised value and its SHAP value
across the 163 explained patients, and a `direction` label. A direction is only
named when |correlation| ≥ 0.1; otherwise the feature is reported as *mixed*, and
a feature nobody in the cohort varies on (an unpopulated one-hot level) is
reported as *undefined* rather than given a spurious sign.

| Task | Feature | mean \|SHAP\| | Corr. | Direction |
| --- | --- | --- | --- | --- |
| ER | ESR1 | 0.2719 | +0.932 | higher expression → toward ER-positive |
| ER | FOXC1 | 0.1547 | −0.872 | higher expression → toward ER-negative |
| ER | MAPT | 0.0638 | +0.868 | → positive |
| ER | NAT1 | 0.0531 | +0.893 | → positive |
| ER | ERBB2 | 0.0399 | −0.701 | → negative |
| ER | stage_III | 0.0369 | −0.879 | → negative |
| PR | ESR1 | 0.1716 | +0.914 | → PR-positive |
| PR | FOXC1 | 0.1350 | −0.870 | → PR-negative |
| PR | CEP55 | 0.0394 | −0.921 | → negative |
| HER2 | FOXC1 | 0.1451 | −0.847 | higher expression → toward HER2-negative |
| HER2 | ANLN | 0.0577 | +0.751 | → HER2-positive |
| HER2 | ESR1 | 0.0565 | −0.638 | → negative |
| HER2 | CEP55 | 0.0390 | +0.895 | → positive |
| Survival | age | 0.4652 | +0.825 | older → **higher** risk |
| Survival | nodal_N0 | 0.3610 | −0.906 | node-negative → **lower** risk |
| Survival | nodal_N2 | 0.1345 | +0.945 | → higher risk |
| Survival | stage_III | 0.1304 | +0.779 | → higher risk |
| Survival | stage_II | 0.1204 | −0.735 | → lower risk |

The receptor directions are the expected ones: ESR1 drives ER and PR positive,
FOXC1 (a basal marker) drives all three receptor outputs negative. The survival
directions are clinically coherent throughout — older age, higher nodal burden
and higher stage all increase predicted risk, node-negative status lowers it.
Coherent direction is a consistency check on the explanation, not evidence that
the survival model is accurate; its test C-index is 0.7112 on 23 events.

### Representative individual patients

For each of ER, PR, HER2 and survival risk the script records three real test
patients — highest, median and lowest explained output — with their full 62-feature
attribution vector in `shap_individual_patients.csv` (540 rows) and a waterfall
plot each. The median is an actual patient index, never an interpolated one.

| Task | Output range across test (low / median / high) | Patients |
| --- | --- | --- |
| ER margin | −0.9512 / 1.5815 / 2.0975 | TCGA-E2-A1LH, TCGA-BH-A1FJ, TCGA-B6-A2IU |
| PR margin | −1.4583 / 0.6709 / 1.0244 | TCGA-A8-A07R, TCGA-EW-A1J6, TCGA-E2-A1BD |
| HER2 margin | −1.5845 / −0.9009 / 0.3999 | TCGA-AO-A1KO, TCGA-AO-A03V, TCGA-A8-A094 |
| Risk score | −2.7011 / −1.2628 / 2.5440 | TCGA-B6-A0RL, TCGA-E2-A1BD, TCGA-D8-A1JF |

The HER2 row shows the default-threshold problem directly: even the *highest*
HER2 margin in the test partition is only +0.3999, and the median patient sits at
−0.9009. The whole distribution is shifted negative, which is why lowering the
threshold to 0.36 (section 7) recovers positives that a 0.5 cut discards.

### Sanity checks

All ten checks required before accepting the results are recorded in
`shap_summary.json` with `passed: true` and supporting evidence:

1. model in eval mode; 2. no optimiser step (0 parameters had a populated
`.grad` after the run — `torch.autograd.grad`, which expected gradients uses,
does not accumulate into `.grad`); 3. checkpoint SHA-256 identical before and
after (`1ad26a25…5baf14`) and parameter-tensor hash identical; 4. re-evaluating
the loaded checkpoint reproduced the recorded test metrics exactly — max absolute
difference 0.000e+00 across 34 metrics; 5. wrapper output identical to the direct
call, and perturbing each block alone measurably moves the output; 6. re-encoding
the same patients with all targets blanked produced a bit-identical feature
matrix, proving no label leaks into the inputs; 7. feature names match the
model's real inputs; 8. all SHAP values finite; 9. SHAP feature width 62 equals
the model input dimensionality; 10. environment recorded (shap 0.52.0,
torch 2.13.0+cpu, numpy 2.5.1, python 3.13.0, Windows-11, CPU).

### Interpreting SHAP values

These are **model-attribution values**: they describe how this trained network's
output responds to its inputs, relative to a training-set background. They are
not causal effects, not evidence of a biological mechanism, not a validated
biomarker ranking, and not a basis for any clinical decision. This limitation is
also recorded in `shap_summary.json` under `interpretation`.

---

## 9. Figures and raw attribution arrays

Under `results/breast_explainability/`:

| Figure | Content |
| --- | --- |
| `shap_er_global.png`, `shap_pr_global.png`, `shap_her2_global.png`, `shap_pam50_global.png`, `shap_survival_global.png` | Top-20 features by mean \|SHAP\|, coloured by modality, alongside the genomic/clinical share |
| `shap_er_beeswarm.png`, `shap_pr_beeswarm.png`, `shap_her2_beeswarm.png`, `shap_survival_beeswarm.png` | Per-patient attribution distributions showing direction, not just magnitude |
| `shap_pam50_beeswarm_{luminal_a,luminal_b,her2_enriched,basal_like,normal_like}.png` | Per-class PAM50 beeswarms |
| `shap_pam50_class_bars.png` | Per-class attribution decomposition across the five subtypes |
| `shap_{er,pr,her2,survival}_patient_{highest,median,lowest}_output.png` | Twelve per-patient waterfall plots for the representative patients in section 8 |

`shap_values.npz` holds the 31 raw arrays behind every figure and table — one
`(163, 62)` attribution matrix per explained output, plus the feature matrix, base
values, predictions and patient IDs. It is **git-ignored**: its `features` array
inverts back to per-patient log1p RSEM expression using the standardisation
statistics stored inside the checkpoint, which makes it patient-level TCGA data
under the same rule that keeps `data/` out of the repository. Regenerate it with
`scripts/run_shap.py` (~17 minutes on CPU); the run is bit-reproducible under
seed 42.

Under `results/breast_analysis/`:

| File | Content |
| --- | --- |
| `detailed_metrics.{json,csv}` | Confusion matrices, positive-class metrics and both PR-AUC conventions for all three runs |
| `threshold_analysis.json` | Validation sweep, selection rule, selected thresholds and their single application to test |
| `threshold_sweep.csv` | All 891 swept operating points (3 runs × 3 tasks × 99 thresholds) |
| `literature_comparison.json` | The verified literature record behind section 11 |
| `predictions/` | **Git-ignored.** Per-patient exported predictions for validation and test, both receptor class probabilities retained. Holds every patient's receptor and subtype labels plus survival time and status, so it is excluded for the same reason as `shap_values.npz`. Regenerate with `scripts/export_predictions.py`. |

Everything aggregate is committed; only the two patient-level intermediates are
not. The representative-patient artefacts (`shap_individual_patients.csv` and the
twelve waterfall plots) *are* committed: they cover 22 patients rather than a
whole partition, and the individual explanation is itself a required result.

---

## 10. Standing limitations

- **PAM50 framing.** The genomic input and the PAM50 target are the same 50-gene
  panel, so the subtype result is *reproduction/recovery of established PAM50
  assignments*, not independent molecular-subtype discovery and not independent
  phenotype prediction. This applies to every run in this document.
- **Single cohort, single split.** All numbers come from one TCGA-BRCA test
  partition of 163 patients under one seed. There is no external cohort, so
  nothing here constitutes external validation, and no claim of clinical utility
  is supported.
- **Survival is thinly powered.** 160 evaluable test patients yield only 23
  events. C-index differences of the magnitude reported here are not
  distinguishable from noise without a test that was not performed.
- **Selection-epoch confound.** The three runs early-stopped at different epochs
  (14 / 28 / 14) under an identical protocol, and `val_total` is Cox-dominated.
  Per-task differences therefore mix the modality effect with the selected epoch.
- **Duration is not directly comparable.** The full run's wall-clock duration was
  not captured in a log, and the two ablations trained for different numbers of
  epochs, so the durations index cost rather than efficiency.
- **Patient-level data is not vendored.** `data/` is git-ignored. The cohort must
  be rebuilt from the sources in `docs/data_sources.md` before any of the
  training commands above will run.
- **SHAP additivity is approximate.** Expected gradients is a sampling estimator,
  so Σ SHAP does not exactly equal prediction − base value. Recorded gaps are
  8–25 % of the standard deviation of the deviation being attributed, so feature
  rankings are approximate and small rank differences should not be read as real.
- **Thresholds were selected on 163 validation patients.** That is a small
  selection set, and the PR result in section 7 shows it directly: a threshold
  with a genuine +0.0315 validation gain lost 0.0198 balanced accuracy on test.
  Reported threshold effects are single-split observations, not tuned operating
  points with any guarantee of transfer.

---

## 11. Literature comparison

`results/breast_analysis/literature_comparison.json` holds the verified record.
Every number below was read from that study's own Europe PMC record — nothing is
reconstructed from memory or from a secondary citation. **Evidence level:
abstract and metadata only**; full texts were not retrieved, so quantities the
abstracts do not state (cohort sizes, validation regime, averaging convention,
event counts) are recorded as unknown rather than inferred.

### Direct comparisons

Same cohort, an overlapping task, a transcriptomic and/or clinical modality, and
a metric on the same scale as ours.

| Study | Dataset | Modality | Model | Task | Metric | Reported | Key methodological difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qiu et al., *Sci Rep* 2026;16:3011 (PMID 41559366) | TCGA-BRCA n=1084; METABRIC n=1980 | Gene expression on PPI + patient-similarity graphs | Multi-task Graph Transformer | 4-class subtype | F1 | **0.872** | **Four classes** — Normal-like excluded; our weakest class. Averaging convention unstated. |
| " | " | " | " | ER / PR / HER2 | ROC-AUC | **0.960 / 0.943 / 0.918** | No clinical modality, so no genomic-vs-clinical ablation is possible. Masking scheme undescribed. |
| " | " | " | " | Overall survival | C-index | **0.721** | Validation regime not stated in the abstract; most plausibly internal CV, vs our single-use held-out test. |
| Babas et al., *Cancers* 2026;18:1497 (PMID 42192859) | TCGA breast (30 % held-out); METABRIC external | MMP/ADAM/ADAMTS protease transcripts + age, stage | Random survival forest | Survival | C-index | **0.797** integrative; **0.742** clinical-only Cox (95 % CI 0.636–0.826); **0.581** METABRIC external | **Endpoint never named**; **no sample sizes reported**, so event counts are unknown. Protease panel, not PAM50. RSF, not a Cox network. |
| **MOGEN-BRCA (this work)** | TCGA-BRCA n=1082, test n=163 | 50 PAM50 genes + 12 clinical | Genomic transformer + clinical MLP + cross-modal attention | 5-class subtype | macro-F1 / acc / ROC-AUC | 0.7904 / 0.8649 / 0.9430 | Five classes; single held-out test scored once |
| " | " | " | " | ER / PR / HER2 | ROC-AUC | 0.9232 / 0.8663 / 0.7736 | Per-task masking; n = 155 / 155 / 134 |
| " | " | " | " | Overall survival | C-index | 0.7112 (23 events) | Clinical-only arm 0.7372; genomics-only 0.6008 |

Reading these honestly: Qiu's receptor AUCs exceed ours on all three tasks, most
sharply on HER2 (0.918 vs 0.7736), and their subtype F1 exceeds our macro-F1 —
though on a strictly easier four-class problem, with an unstated averaging
convention and an unstated validation regime. Our survival C-index (0.7112) is
close to theirs (0.721). The one thing our study provides that neither direct
comparison does is the **measured** decomposition of which modality carries which
task.

Babas's clinical-only C-index of 0.742 is strikingly close to our clinical-only
0.7372, which is reassuring for our clinical branch. The directions then diverge:
adding transcripts raised theirs to 0.797, while our multimodal model (0.7112)
scored *below* clinical-only. Their clinical-only interval (0.636–0.826) is wide
enough to contain both figures, which is an argument against over-reading either
gap.

### Contextual comparisons

Shared metric or shared experimental shape, but a different dataset, modality or
endpoint. These frame the result; they do not benchmark it.

| Study | Dataset | Modality | Task | Metric | Reported | Why contextual only |
| --- | --- | --- | --- | --- | --- | --- |
| Lyu et al., iMCN, *Biomed Phys Eng Express* 2026 (PMID 41564441) | TCGA-BRCA, TCGA-LUAD | WSI + genomics | Survival | C-index | 0.740 BRCA; 0.691 LUAD | Histopathology imaging is outside this architecture. A ceiling reference for TCGA-BRCA survival, not like-for-like. |
| Wang et al., MuTriM, *Eur J Cancer* 2026;238:116679 (PMID 41850009) | FUSCC n=335 train; TCGA n=126 external | DCE-MRI radiomics + WSI pathomics | **RFS**, not OS | C-index | **0.75** multimodal vs **0.65** MRI-only vs **0.70** WSI-only | Different modalities and endpoint, not trained on TCGA. Included because it is the closest published breast multimodal-vs-unimodal ablation. |
| Cheerla & Gevaert, *Bioinformatics* 2019;35:i446 (PMID 31510656) | 20 cancer types | Clinical + mRNA + miRNA + WSI, multimodal dropout | Prognosis | C-index | 0.78 pooled | Pan-cancer pooled and includes WSI; no breast-specific value in the record. |
| de Negreiros Botan & de Sousa, Core-PAM50, *Breast Cancer Res* 2026 (PMID 42169063) | METABRIC n=2173 train; **TCGA-BRCA n=1098 external**; GSE25066 n=508 | Expression + clinical | Overall survival | C-index | METABRIC 0.584; **TCGA external ≈0.42**; TCGA OS HR 0.89 (0.67–1.19) | Externally trained. Its value is as evidence about the endpoint, not as a benchmark. |
| Uma Kandan & Abul, *Comput Methods Programs Biomed* 2026;285:109510 (PMID 42302593) | SCAN-B train; TCGA-BRCA external | Within-sample **rank order** of PAM50 genes | Subtype | precision/recall; accuracy | ≥91 % on SCAN-B; "at least 95 % accuracy" on TCGA for **"the majority of subtypes"** | A per-subtype floor over a favourable subset, class count and cohort sizes unstated, and a transfer result. Not comparable to our all-class 0.8649. |

Two of these matter more than the rest.

**MuTriM is the closest published version of our own ablation experiment**, and it
found the opposite ordering: multimodal beat both unimodal arms by +0.10 and
+0.05. Multimodal superiority is therefore not a universal result even within
breast cancer, and our measured ordering (clinical-only > multimodal >
genomics-only on survival) should not be presented as a failure to reproduce an
established law.

**Core-PAM50 is independent external support that TCGA-BRCA overall survival is a
low-signal endpoint.** An independent group took a PAM50-derived expression
signature into TCGA-BRCA OS on essentially our cohort (n=1098 vs our 1082) and
obtained a C-index of about 0.42 — below chance — with a hazard ratio of 0.89
(0.67–1.19) crossing the null, which they attribute to "short follow-up and low
event rates". Our test partition has 23 events among 160 evaluable patients and a
median follow-up of 28.5 months. Our genomics-only C-index of 0.6008 is therefore
consistent with the endpoint's difficulty and is not, on this evidence, an
implementation defect.

### Methodologically relevant, not numerically comparable

Borges, *JAMIA Open* 2026;9(2):ooaf177 (PMID 41937807) predicts 4-class PAM50 on
TCGA-BRCA (691 complete cases) from **ER status, PR status, HER2 status, tumour
stage and age**. It reports McFadden pseudo-R² 0.396 and ΔP up to +0.29 for
HER2-enriched, but no accuracy, AUC, F1 or confusion matrix, so nothing in it can
be placed beside our clinical-only PAM50 accuracy of 0.4932.

It matters anyway, for two reasons. First, it feeds receptor status in as a
*predictor* of subtype — precisely what this project's data contract forbids, and
its own framing is that such a model "effectively substitutes biomarker status for
molecular data". Second, it is the mirror image of our clinical-only result:
Borges shows clinical variables *including* receptor status retain moderate
subtype explanatory power, whereas our clinical-only arm, which excludes receptor
status and uses only age, stage, nodal stage and sex, reaches PAM50 ROC-AUC 0.4505
and predicts three of five classes. Read together, the apparent subtype signal in
clinical-only models appears to come largely from receptor status rather than from
age and stage.

### Studies inspected and excluded

Two frequently cited multimodal-survival methods were inspected and **cannot be
cited numerically**: SALMON (Huang et al., *Front Genet* 2019;10:166, PMID
30906311) and MultiSurv (Vale-Silva & Rohr, *Sci Rep* 2021;11:13505, PMID
34188098). Neither abstract reports a numeric C-index, and full texts were not
retrieved, so quoting a figure for either would mean inventing one. Also excluded:
several TCGA-BRCA signature papers reporting C-indices of 0.858–0.92, all of which
are within-subtype, nomogram-calibrated, or derived on synthetic data without
external OS validation.

### Limits of this comparison

- No study here shares our exact protocol, so **no entry is a strict benchmark**.
  Differences in class count, endpoint, cohort size, validation regime and
  masking are individually large enough to move a metric by more than the gaps
  being compared.
- Evidence is **abstract-level**. Where an abstract omits the validation regime or
  the cohort size, that omission is recorded, not filled in.
- Our numbers come from a **single test partition scored once**. Most reported
  literature values are cross-validated or drawn from the best of several
  configurations, which biases the comparison against us in an unquantified way.
- No confidence interval or significance test was computed for any MOGEN-BRCA
  metric, so none of these comparisons is a statistical claim.
- Metric names are not always definitions. Our own `pr_auc` for receptor tasks is
  a macro over both classes (section 4), and F1 averaging conventions in the cited
  work are frequently unstated.
