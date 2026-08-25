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

---

## 1. Reproducing the runs

```bash
uv run pytest                                     # 276 tests

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
```

`scripts/run_ablation_comparison.py` and `scripts/run_shap.py` never write into
`results/breast/`; the SHAP script additionally re-hashes the frozen checkpoint
before and after the analysis and fails if the two hashes differ.

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

## 6. SHAP attribution for the frozen model

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
`results/breast_explainability/shap_feature_importance.csv` (620 rows).

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

## 7. Figures

Under `results/breast_explainability/`:

| Figure | Content |
| --- | --- |
| `shap_er_global.png`, `shap_pr_global.png`, `shap_her2_global.png`, `shap_pam50_global.png`, `shap_survival_global.png` | Top-20 features by mean \|SHAP\|, coloured by modality, alongside the genomic/clinical share |
| `shap_er_beeswarm.png`, `shap_pr_beeswarm.png`, `shap_her2_beeswarm.png`, `shap_survival_beeswarm.png` | Per-patient attribution distributions showing direction, not just magnitude |
| `shap_pam50_beeswarm_{luminal_a,luminal_b,her2_enriched,basal_like,normal_like}.png` | Per-class PAM50 beeswarms |
| `shap_pam50_class_bars.png` | Per-class attribution decomposition across the five subtypes |

---

## 8. Standing limitations

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
