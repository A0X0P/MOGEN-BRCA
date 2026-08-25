# src/CLAUDE.md

## AUTHORITATIVE PROJECT ARCHITECTURE

This project is exclusively a TCGA-BRCA breast-cancer model using TWO modalities:

1. Gene Expression
2. Clinical Data

There is NO imaging modality in the current architecture.

Do NOT implement, restore, prioritize, or introduce image/histopathology
processing into the active BRCA pipeline.

The authoritative architecture is:

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



## ACTIVE DATA CONTRACT

Cohort:
- TCGA-BRCA
- N = 1082 retained patient intersection before per-task masking

Genomic input:
- 50 PAM50 genes
- RSEM gene-expression values
- log1p transformation
- train-fold standardization
- fixed deterministic gene ordering

Clinical input:
- age
- pathological tumour stage
- pathological nodal stage
- sex

Clinical representation:
- 12-dimensional baseline vector:
  age
  + stage I/II/III/IV/Unknown
  + nodal N0/N1/N2/N3/Unknown
  + sex

Targets:
- PAM50 subtype: Luminal A, Luminal B, HER2-enriched,
  Basal-like, Normal-like
- ER status
- PR status
- HER2 status
- overall-survival risk

Missing labels:
- Per-task masking.
- A patient participates in a task only when that task's target
  is available.
- Do NOT require complete-case data across all tasks.

Survival:
- DeepSurv/Cox partial-likelihood objective
- OS_MONTHS and OS_STATUS
- OS == 0 censored cases excluded from survival loss only
- classification data retained

Architecture:
- Genomic Transformer
- Clinical MLP
- Cross-Modal Attention
- Concatenation
- Fusion MLP
- PAM50 classification head
- ER/PR/HER2 receptor heads
- DeepSurv survival head

## IMPLEMENTATION DIRECTIVE

The architecture and data contract above are FINAL for the current
TCGA-BRCA implementation.

STOP AUDITING THE ARCHITECTURE.

Do not ask for further methodological approval unless an actual
implementation blocker is encountered.

Implement the active BRCA pipeline now.

Priority order:

1. BRCA dataset loading and patient alignment
2. PAM50 gene extraction and preprocessing
3. Clinical feature preprocessing
4. Per-task label masking
5. Two-modality model construction
6. Masked multitask classification loss
7. DeepSurv/Cox survival loss
8. BRCA training configuration
9. Training execution
10. Evaluation and result generation
11. Tests and import/static checks

The first objective is to obtain a REAL end-to-end TCGA-BRCA training
run. Do not spend extended time searching for alternative datasets,
PAM50 gene lists, papers, or methodological alternatives unless an
actual implementation blocker requires it.

## IMAGING CODE

The following are legacy/inactive components:

src/data/datasets/imaging_2d_dataset.py
src/data/datasets/imaging_3d_dataset.py
src/data/preprocessing/imaging_preprocess.py
src/data/schema/imagery.py
src/models/imaging/encoder_2d.py
src/models/imaging/encoder_3d.py

They must NOT be imported or used by the active BRCA pipeline.

Before deleting any of these files, search the repository for imports
and references to them. If they are unused, they may be removed.
If they are referenced by tests or unrelated legacy code, report the
references and do not blindly delete them.

multimodal_dataset.py must represent the CURRENT two-modality design,
not the previous imaging/genomics/clinical design.

model_factory.py must construct ONLY:

- Genomic Transformer
- Clinical MLP
- Cross-Modal Attention
- Fusion MLP
- PAM50 subtype head
- ER head
- PR head
- HER2 head
- DeepSurv survival head

No image encoder or image branch may be instantiated.

## TRAINING REQUIREMENT

After implementation, do not stop at "implementation complete".

Run an actual TCGA-BRCA training experiment using the available data.

The minimum successful milestone is:

- data loads successfully
- patient IDs align
- PAM50 features load successfully
- clinical features load successfully
- labels and masks are generated
- model instantiates successfully
- forward pass succeeds
- all task losses compute successfully
- backward pass succeeds
- optimizer step succeeds
- validation runs
- checkpoints/results are written
- at least one complete training run finishes

If a failure occurs, diagnose and fix the implementation rather than
returning to the architecture audit.

Report:

1. files changed
2. files deleted
3. files created
4. remaining imaging references
5. dataset dimensions
6. train/validation/test sizes
7. model parameter count
8. training configuration
9. training/validation losses
10. classification metrics
11. survival metrics
12. location of saved checkpoint and results