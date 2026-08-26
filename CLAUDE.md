# CLAUDE.md

# Cancer Framework — Current Project Instructions

This file gives Claude Code the authoritative context for working in this
repository.

The current active implementation and submission target is:

**Breast Cancer Molecular Subtype and Receptor-Status Classification, and
Survival Prognosis Using Gene Expression and Clinical Data**

The active implementation uses **TCGA-BRCA** and exactly **two modalities**:

1. Gene Expression
2. Structured Clinical Data

The previous three-modality architecture involving medical imaging is
**legacy architecture and is NOT part of the active BRCA system**.

---

# 1. AUTHORITATIVE CURRENT SCOPE

## Active Cancer

The current implementation is exclusively:

**TCGA-BRCA — Breast Cancer**

Do not expand the active implementation to lung, prostate, or other cancers
unless explicitly instructed.

The repository may contain legacy scaffolding for other cancer types, but
that code must not determine the architecture of the active BRCA pipeline.

## Active Modalities

The active BRCA pipeline has exactly two modalities:

- Gene Expression
- Clinical Data

There is **NO imaging modality** in the current BRCA architecture.

Do not introduce or restore:

- histopathology images
- mammograms
- CT
- MRI
- PNG/JPEG image pipelines
- DICOM processing
- 2D CNN encoders
- 3D CNN encoders
- EfficientNet
- image transformers
- image preprocessing
- image schemas
- image branches in the model factory

Existing imaging code is considered legacy/inactive unless explicitly
requested otherwise.

---

# 2. AUTHORITATIVE BRCA ARCHITECTURE

The current model architecture is:

```text
                  TCGA-BRCA Patient
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ↓                           ↓
       Gene Expression              Clinical Data
             │                           │
             ↓                           ↓
    Genomic Transformer             Clinical MLP
             │                           │
             ↓                           ↓
      Genomic Embedding            Clinical Embedding
             │                           │
             └─────────────┬─────────────┘
                           │
                           ↓
                 Cross-Modal Attention
                           │
             ┌─────────────┴─────────────┐
             ↓                           ↓
       Genomic latent              Clinical latent
             │                           │
             └─────────────┬─────────────┘
                           │
                           ↓
                    Concatenation
                           │
                           ↓
                       Fusion MLP
                           │
                           ↓
                  Fused Representation
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ↓                  ↓                  ↓
    PAM50 Head       Receptor Heads       DeepSurv
        │             │    │    │             │
        ↓             ↓    ↓    ↓             ↓
      5-class         ER   PR  HER2       Risk Score

The active model therefore consists of:

Gene Expression
    ↓
Genomic Transformer
    ↓
Genomic representation

Clinical Data
    ↓
Clinical MLP
    ↓
Clinical representation

Genomic + Clinical representations
    ↓
Cross-Modal Attention
    ↓
Concatenation
    ↓
Fusion MLP
    ↓
Fused representation
    ├── PAM50 subtype head
    ├── ER head
    ├── PR head
    ├── HER2 head
    └── DeepSurv survival head
3. ACTIVE RESEARCH TASKS

The active BRCA system performs five prediction tasks.

3.1 PAM50 Molecular Subtype

Five classes:

Luminal A
Luminal B
HER2-enriched
Basal-like
Normal-like

The subtype target represents the existing PAM50 classification in the
TCGA-BRCA clinical data.

This task should be described scientifically as PAM50 subtype
classification/reproduction, not independent discovery of molecular
subtypes.

3.2 Receptor Status

Three independent binary classification tasks:

ER
PR
HER2

The receptor targets must never be derived from the PAM50 subtype label.

3.3 Overall Survival

The survival task predicts overall-survival risk using a DeepSurv/Cox
proportional-hazards formulation.

The survival target consists of:

OS_MONTHS
OS_STATUS

No progression-free survival, disease-free survival, or other survival
endpoint is part of the current primary task.

4. ACTIVE DATA CONTRACT
Cohort

The current validated TCGA-BRCA cohort is:

N = 1082 patients

This is the retained three-way patient intersection established during the
data validation process.

The cohort is retained using per-task label masking rather than
complete-case filtering.

Genomic Input

The genomic input consists of the 50 PAM50 genes.

The canonical gene set is:

ACTR3B
ANLN
BAG1
BCL2
BIRC5
BLVRA
CCNB1
CCNE1
CDC20
CDC6
CDH3
CENPF
CEP55
CXXC5
EGFR
ERBB2
ESR1
EXO1
FGFR4
FOXA1
FOXC1
GPR160
GRB7
KIF2C
KRT14
KRT17
KRT5
MAPT
MDM2
MELK
MIA
MKI67
MLPH
MMP11
MYBL2
MYC
NAT1
NDC80
NUF2
ORC6L
PGR
PHGDH
PTTG1
RRM2
SFRP1
SLC39A6
TMEM45B
TYMS
UBE2C
UBE2T

The published-symbol resolution includes:

KNTC2 → NDC80
CDCA1 → NUF2

The gene ordering must remain deterministic and documented.

Genomic Preprocessing

The intended genomic preprocessing is:

RSEM expression
    ↓
log1p transformation
    ↓
train-fold gene-wise standardization
    ↓
PAM50 genomic tensor
    ↓
Genomic Transformer

Do not fit normalization statistics using validation or test data.

5. CLINICAL INPUT CONTRACT

The baseline clinical representation is 12 dimensions:

1 × age

5 × pathological tumour stage
    I
    II
    III
    IV
    Unknown

5 × pathological nodal stage
    N0
    N1
    N2
    N3
    Unknown

1 × sex

Total:

1 + 5 + 5 + 1 = 12
Clinical Variables

Use:

age
pathological tumour stage
pathological nodal stage
sex

Do not introduce unavailable clinical variables merely to match an older
architecture.

In particular, the old clinical contract involving:

weight
height
BMI
smoking

must not be assumed to apply to BRCA.

6. LABEL COUNTS AND MASKING

The validated usable-label counts are approximately:

PAM50 subtype    981
ER              1031
PR              1028
HER2             937
Survival        1069

These counts are task-specific.

Do NOT require every patient to have every target.

A patient participates in each task only when that task's target is
available.

The model and loss functions must therefore support per-task masking.

7. HER2 LABEL POLICY

HER2 labels use the validated score-driven rule:

IHC score 3
    → POSITIVE

IHC score 0 or 1
    → NEGATIVE

IHC score 2
    → resolve using FISH

Equivocal/Indeterminate IHC
    → resolve using FISH where available

Definitive IHC/FISH conflict
    → FISH wins

No usable evidence
    → masked

This produces:

937 usable HER2 labels
769 NEGATIVE
168 POSITIVE

Do not replace this with a naive summary-status-only rule.

8. SURVIVAL POLICY

The survival task uses overall survival from the PanCancer clinical source.

Patients with:

OS_MONTHS == 0

are excluded from the survival loss only when they are censored observations.

They remain available for classification tasks.

Do not introduce an artificial epsilon solely to convert zero survival time
into a positive value.

The unresolved survival conflict involving TCGA-E9-A245 is excluded from
the survival task but retained for classification.

9. MODEL COMPONENTS

The active model factory must construct only:

Genomic Transformer
Clinical MLP
Cross-Modal Attention
Fusion MLP
PAM50 classification head
ER classification head
PR classification head
HER2 classification head
DeepSurv survival head

src/models/model_factory.py must not instantiate imaging components.

The genomic encoder operates on the PAM50 gene-expression representation.

The clinical encoder projects the 12-dimensional clinical vector into the
embedding space required by the fusion mechanism.

Cross-modal attention must explicitly handle the dimensional difference
between the genomic token sequence and the clinical representation.

The resulting representations are combined through the fusion stage before
being passed to the task-specific heads.

10. ACTIVE DATA PIPELINE

The active BRCA pipeline should conceptually follow:

TCGA-BRCA raw data
        │
        ├── RNA-seq
        │      ↓
        │   PAM50 extraction
        │      ↓
        │   preprocessing
        │
        └── Clinical data
               ↓
          clinical feature extraction
               ↓
          clinical preprocessing
               │
               └──────────────┐
                              ↓
                     Patient alignment
                              ↓
                         Dataset
                              ↓
                     Two-modality model
                              ↓
                         Multi-task loss
                              ↓
                          Training
                              ↓
                        Evaluation

Patient identifiers must be used to align genomic and clinical records.

Never align patients by row position unless the source explicitly
guarantees identical ordering.

11. DATA LEAKAGE PREVENTION

The following are mandatory:

Do not use target labels as model inputs.
Do not derive ER/PR/HER2 labels from PAM50 subtype.
Do not derive HER2 labels from the subtype label.
Do not use survival outcomes as clinical input features.
Do not fit normalization statistics on validation/test data.
Do not allow patient overlap between train/validation/test partitions.
Split at the patient level.
Maintain deterministic splits through the configured random seed.

The clinical feature loader should use an explicit allow-list of permitted
clinical features.

Do not blindly load all columns and "drop obvious labels."

12. DATA SPLITTING

All splits must be performed at the patient level.

The same patient must never occur in:

train
validation
test

simultaneously.

Preprocessing statistics such as means and standard deviations must be
learned from the training partition only.

The validation and test partitions must use the training statistics.

13. MISSING DATA

Missing feature values must be explicitly represented and handled.

Missing target labels must be handled through task-specific masks.

Do not silently replace missing labels with arbitrary classes.

Do not convert missing survival outcomes into artificial events or times.

14. LEGACY IMAGING CODE

The repository may contain files from the previous imaging-based
architecture, including:

src/data/datasets/imaging_2d_dataset.py
src/data/datasets/imaging_3d_dataset.py
src/data/preprocessing/imaging_preprocess.py
src/data/schema/imagery.py
src/models/imaging/encoder_2d.py
src/models/imaging/encoder_3d.py

These are legacy components.

They must NOT be imported or used by the active TCGA-BRCA pipeline.

Before deleting any legacy file:

Search the repository for imports and references.
Determine whether it is used by the active BRCA pipeline.
Remove obsolete references.
Run tests/import checks.
Delete the file only when safe.

Do not perform destructive cleanup merely for aesthetic reasons.

The immediate priority is a functioning BRCA training pipeline.

15. LEGACY MULTI-CANCER ARCHITECTURE

Earlier versions of this repository described:

Breast + Lung + Prostate

and:

Imaging + Genomics + Clinical

That architecture is historical.

It must not override the current BRCA implementation.

Legacy code can remain temporarily if it does not interfere with the active
BRCA pipeline.

Do not spend time restoring or completing the old lung/imaging architecture
unless explicitly instructed.

16. PROJECT STRUCTURE

The current active structure includes:

configs/
    breast/

data/
    raw/
    interim/
    processed/
    metadata/

src/
    data/
        datasets/
        preprocessing/
        schema/

    models/
        genomics/
        tabular/
        fusion/
        classification/
        survival/

    training/

    evaluation/

    inference/

    utils/

scripts/

tests/

notebooks/

The active BRCA implementation should use the existing repository structure
rather than introducing unnecessary new directory hierarchies.

17. CONFIGURATION

Use YAML configuration for:

dataset paths
preprocessing settings
model dimensions
hidden dimensions
attention dimensions
dropout
optimizer
learning rate
batch size
epochs
random seed
device
loss weights
checkpoint paths
evaluation settings

Do not hardcode experiment-specific values into model or training code.

18. DEVICE HANDLING

The framework must remain device-agnostic.

Resolve the device once from configuration.

Conceptually:

device = torch.device(
    "cuda" if cfg.device == "auto" and torch.cuda.is_available()
    else cfg.device
)

Use:

model.to(device)
tensor.to(device)

Do not use .cuda() or .cpu() directly inside the active model,
training, dataset, or inference implementation.

Mixed precision must only be enabled when CUDA is actually available.

The current environment may be CPU-only. The implementation must therefore
be capable of completing a real training run on CPU.

19. REPRODUCIBILITY

Every training run must use a deterministic seed where practical.

Record:

random seed
configuration
dataset information
model configuration
package/environment information where practical
git commit hash when available
training metrics
validation metrics
checkpoint information

Do not modify random seeds during a training run.

20. TRAINING

The training system must support:

multi-task classification
task-specific label masking
DeepSurv/Cox loss
validation
checkpointing
early stopping where configured
reproducible execution
CPU execution
optional CUDA execution
resume training where supported

The critical requirement for the current project is an actual successful
TCGA-BRCA training run.

A successful implementation is NOT complete merely because the code imports.

At minimum, verify:

Dataset loads.
Patient IDs align.
PAM50 features load.
Clinical features load.
Labels load.
Task masks are generated.
Model instantiates.
Forward pass succeeds.
All applicable task losses compute.
Backward pass succeeds.
Optimizer step succeeds.
Validation executes.
Checkpoint is saved.
Evaluation produces metrics.
21. EVALUATION

Classification metrics should include where applicable:

Accuracy
Precision
Recall
F1
Macro-F1
ROC-AUC
PR-AUC

For the multi-class PAM50 task, use appropriate multi-class averaging.

For survival:

Concordance Index
Time-dependent AUC when implemented and supported by the evaluation
pipeline

Do not report metrics for a task using unavailable labels.

Evaluation must respect task-specific masks.

22. INTERPRETABILITY

Interpretability is secondary to obtaining a correct training and evaluation
pipeline.

For the current BRCA system, interpretation should focus on:

genomic feature importance
attention analysis
clinical feature importance
model-level explanations where practical

Do not introduce Grad-CAM or other image-specific explainability into the
active BRCA pipeline.

23. LOGGING

Use structured logging.

Do not use print() for normal application/training logging.

Training logs should include:

experiment configuration
epoch
training loss
validation loss
task-specific metrics
learning rate where useful
checkpoint events
dataset information
24. TESTING

Every major module should have unit tests.

Critical paths should have integration tests.

At minimum, test:

dataset loading
patient alignment
PAM50 extraction
clinical preprocessing
label masking
model construction
forward pass
loss computation
training step
checkpoint save/load
evaluation

Tests must reflect the current two-modality architecture.

Do not write tests that require imaging for the active BRCA pipeline.

25. CODING STANDARDS

Use:

snake_case       variables/functions
PascalCase       classes
UPPER_CASE       constants

Every public function should have:

type hints
a useful docstring

Prefer:

small focused functions
shallow nesting
early returns
composition
explicit data flow
meaningful exceptions

Avoid:

duplicated logic
silent failures
unnecessary abstractions
premature optimization
unexplained magic numbers
broad Any usage when a concrete type is possible

Functions should generally remain below approximately 50 lines unless
complexity clearly justifies otherwise.

26. ERROR HANDLING

Raise meaningful exceptions.

Do not silently swallow errors.

Validate:

configuration
file paths
tensor shapes
patient identifiers
label vocabularies
feature dimensions
model dimensions

Shape mismatches should fail clearly rather than being silently reshaped.

27. DOCUMENTATION

Documentation must describe the CURRENT architecture.

Do not add new documentation claiming that the active BRCA system uses
medical imaging.

When modifying documentation, prefer the current:

Gene Expression + Clinical Data

architecture.

The academic interpretation must remain scientifically accurate.

28. DEVELOPMENT PRIORITY

The current deadline requires prioritizing a working experimental pipeline.

Use this order:

1. Dataset
2. Preprocessing
3. Patient alignment
4. Label masking
5. Model
6. Losses
7. Training
8. Evaluation
9. Checkpoint/results
10. Tests
11. Documentation refinement

Do not spend extended time on architectural audits once the architecture has
been established.

Do not search for alternative datasets unless the currently available
TCGA-BRCA data presents a genuine implementation blocker.

Do not repeatedly re-validate decisions that have already been established
in the active data contract.

If an actual blocker is encountered, diagnose it and fix it.

29. CURRENT VALIDATED DATA SUMMARY

The current validated cohort specification is:

Cohort:
    TCGA-BRCA
    N = 1082

PAM50:
    50 genes
    981 usable subtype labels

ER:
    1031 usable labels

PR:
    1028 usable labels

HER2:
    937 usable labels

Overall Survival:
    1069 usable survival observations
    151 deaths

The active clinical vector is:

12 dimensions

The active genomic vector is:

50 PAM50 genes

The model therefore operates on:

50-gene expression representation
+
12-dimensional clinical representation

followed by cross-modal fusion and multi-task prediction.

30. CURRENT IMPLEMENTATION RULE

The architecture and data contract in this document are FINAL for the
current TCGA-BRCA implementation unless the user explicitly changes them.

Claude Code should not repeatedly ask for methodological approval on these
already-established decisions.

Do not return to the old imaging architecture.

Do not add a third modality.

Do not invent additional clinical variables.

Do not replace the PAM50 target with a different subtype system.

Do not replace overall survival with another survival endpoint.

Do not replace task-specific masking with complete-case filtering without
explicit instruction.

31. COMMANDS

Typical commands should follow the repository's current uv setup.

Examples:

uv run pytest

Training should use the active BRCA configuration, for example:

uv run scripts/run_train.py --config configs/breast/train.yaml

Evaluation should use the resulting checkpoint:

uv run scripts/run_eval.py \
    --config configs/breast/train.yaml \
    --checkpoint <checkpoint>

If the actual repository uses different script/config paths, inspect the
existing implementation rather than inventing new paths.