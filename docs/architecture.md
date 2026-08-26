# Architecture

## Overview

This framework is a multimodal deep learning system for **breast cancer molecular subtype classification, receptor-status classification, and survival prognosis** using matched **gene expression and clinical data** from the **TCGA-BRCA cohort**.

The architecture contains two input modalities:

1. **Gene expression data**
2. **Structured clinical/pathological data**

The genomic modality is encoded using a **Genomic Transformer**, while the clinical modality is encoded using a **Clinical MLP**. The resulting modality-specific embeddings are integrated using **cross-modal attention**, followed by concatenation and a **fusion MLP** that produces a shared patient representation.

The fused representation is passed simultaneously to multiple task-specific prediction heads:

* **PAM50 molecular subtype classification** — five classes
* **ER receptor-status classification** — binary
* **PR receptor-status classification** — binary
* **HER2 receptor-status classification** — binary
* **Survival prognosis** — DeepSurv/Cox proportional hazards risk prediction

The system therefore follows a **shared multimodal representation with multi-task prediction** design.

The architecture is specific to **TCGA-BRCA** and does not attempt to provide a generic multi-cancer or multi-modality framework.

---

## High-Level Data Flow

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
                  │                           │
                  ↓                           ↓
           Genomic Latent               Clinical Latent
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
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ↓                   ↓                   ↓
       PAM50 Head         Receptor Heads       DeepSurv Head
            │             ┌────┼────┐                │
            │             │    │    │                │
            ↓             ↓    ↓    ↓                ↓
       5-class            ER   PR  HER2          Risk Score
```

---

## Data Sources and Scope

The framework operates on the **TCGA-BRCA** breast cancer cohort.

### Genomic Modality

The genomic modality consists of bulk RNA-seq gene expression measurements.

The primary genomic representation is restricted to the **PAM50 gene set**, which is used for molecular intrinsic subtype classification.

The five PAM50 molecular subtypes are:

1. Luminal A
2. Luminal B
3. HER2-enriched
4. Basal-like
5. Normal-like

The genomic preprocessing pipeline is responsible for:

* Loading gene expression measurements.
* Selecting the required PAM50 genes.
* Aligning gene identifiers.
* Handling missing or invalid expression values.
* Applying the configured normalization/transformation.
* Producing a consistent feature vector for every eligible patient.

The genomic feature ordering must remain consistent between preprocessing, dataset construction, training, evaluation, and inference.

### Clinical Modality

The clinical modality consists of structured clinical and pathological variables available for the TCGA-BRCA cohort.

Candidate variables include:

* Age at diagnosis
* Tumour stage
* Nodal status
* Other selected pathological/clinical variables available consistently across the cohort

Clinical preprocessing is responsible for:

* Selecting the configured clinical variables.
* Handling missing values.
* Encoding categorical variables.
* Normalizing/standardizing continuous variables where required.
* Producing a fixed-dimensional clinical feature vector.

Demographic variables such as race/ethnicity should not automatically be treated as primary predictive features. When included, they should be handled according to the experimental design and may be reserved for cohort characterization and subgroup analysis.

---

## Data Contracts

Data entering the modeling pipeline should conform to the schemas defined under:

```text
src/data/schema/
```

The core patient-level representation should contain the information required to construct the two model modalities and the prediction targets.

A conceptual patient record is:

```text
Patient
├── patient_id
├── gene_expression
├── clinical
└── targets
    ├── pam50_subtype
    ├── er_status
    ├── pr_status
    ├── her2_status
    ├── survival_time
    └── event
```

### Gene Expression Data

The genomic data contract should contain:

* Patient identifier
* Gene identifiers
* Corresponding expression values
* PAM50 feature alignment information

### Clinical Data

The clinical data contract should contain:

* Patient identifier
* Configured clinical/pathological variables
* Encoded or raw values according to the preprocessing stage

### Survival Data

The survival target consists of:

* Observed survival time
* Event/censoring indicator

The survival representation follows the requirements of the Cox proportional hazards objective used by the DeepSurv head.

---

## Dataset Construction

The dataset layer converts validated and preprocessed patient-level records into tensors suitable for PyTorch.

The primary dataset is responsible for returning:

```text
{
    genomic_features,
    clinical_features,
    pam50_target,
    er_target,
    pr_target,
    her2_target,
    survival_time,
    event
}
```

The exact structure may vary according to the implementation, but the semantic contract should remain stable.

Patient matching is performed using the patient identifier so that gene expression, clinical variables, classification labels, and survival information correspond to the same patient.

Data leakage must be prevented during preprocessing and feature transformation. Operations that estimate parameters from data, such as normalization or imputation statistics, should be fitted using the training partition only and then applied to validation/test data.

---

# Model Architecture

The model is assembled through:

```text
src/models/model_factory.py
```

The model consists of four major stages:

1. Modality-specific encoding
2. Cross-modal interaction
3. Multimodal fusion
4. Multi-task prediction

---

## 1. Genomic Encoder

### Module

```text
src/models/genomics/genomic_transformer.py
```

The Genomic Transformer converts the PAM50 gene expression vector into a learned genomic representation.

Conceptually:

```text
PAM50 Expression Vector
        │
        ↓
Feature Projection / Embedding
        │
        ↓
Transformer Encoder
        │
        ↓
Pooling / Representation Extraction
        │
        ↓
Genomic Embedding
```

The Transformer is used to model relationships between genomic features rather than treating every gene independently.

The encoder produces a fixed-dimensional representation:

```text
x_g → h_g
```

where:

* `x_g` = genomic input
* `h_g` = genomic embedding

The dimensionality of `h_g` is determined by the model configuration.

---

## 2. Clinical Encoder

### Module

```text
src/models/tabular/clinical_mlp.py
```

The Clinical MLP transforms structured clinical features into a learned clinical representation.

Conceptually:

```text
Clinical Features
       │
       ↓
Input Projection
       │
       ↓
MLP / Residual Blocks
       │
       ↓
Clinical Embedding
```

The clinical encoder produces:

```text
x_c → h_c
```

where:

* `x_c` = clinical input
* `h_c` = clinical embedding

The resulting clinical embedding has a dimensionality compatible with the multimodal fusion stage.

---

# 3. Cross-Modal Attention

### Module

```text
src/models/fusion/cross_modal_attention.py
```

Cross-modal attention models interactions between the genomic and clinical representations.

Rather than immediately concatenating the two embeddings, the architecture allows information from one modality to influence the representation of the other.

Conceptually:

```text
Genomic Embedding ──────┐
                        │
                        ↓
                Cross-Modal Attention
                        ▲
                        │
Clinical Embedding ─────┘
```

The attention mechanism produces modality-aware latent representations:

```text
h_g, h_c
   │
   ↓
Cross-Modal Attention
   │
   ├── h'_g
   └── h'_c
```

where:

* `h'_g` = attention-enhanced genomic latent representation
* `h'_c` = attention-enhanced clinical latent representation

The precise query/key/value arrangement is implementation-dependent, but the purpose of this module is to explicitly model cross-modal dependencies before final fusion.

---

# 4. Multimodal Fusion

After cross-modal attention, the resulting modality-specific latent representations are concatenated:

```text
h'_g ─────┐
          │
          ├── Concatenation → Fusion MLP → z
          │
h'_c ─────┘
```

### Fusion Module

```text
src/models/fusion/fusion_head.py
```

The concatenated representation is processed by a Fusion MLP.

Formally:

```text
h_f = MLP([h'_g || h'_c])
```

where:

* `||` denotes vector concatenation
* `h_f` is the final fused patient representation

The fused representation `h_f` is shared by all downstream prediction heads.

This architecture deliberately separates:

* **cross-modal interaction** — Cross-Modal Attention
* **representation integration** — Concatenation
* **nonlinear multimodal transformation** — Fusion MLP

This provides a clear intermediate-fusion architecture rather than treating concatenation itself as the complete fusion mechanism.

---

# Prediction Heads

The final fused representation feeds multiple task-specific prediction heads.

```text
                    Fused Representation
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ↓                 ↓                 ↓
     PAM50 Head       Receptor Heads      DeepSurv Head
          │             │    │    │             │
          ↓             ↓    ↓    ↓             ↓
      5 Classes          ER   PR  HER2       Risk Score
```

The architecture is therefore multi-task: the encoder and fusion layers are shared, while each prediction objective has its own output head.

---

## PAM50 Molecular Subtype Head

The PAM50 classification head predicts one of five intrinsic molecular subtypes:

```text
Luminal A
Luminal B
HER2-enriched
Basal-like
Normal-like
```

The head produces logits:

```text
z_pam50 ∈ R^5
```

followed by a softmax operation during inference to obtain class probabilities.

Training uses an appropriate multi-class classification loss, typically cross-entropy.

---

## Receptor-Status Heads

Three independent receptor-status prediction heads are used:

```text
ER Head
PR Head
HER2 Head
```

Each head performs binary classification.

Conceptually:

```text
Fused Representation
       │
       ├── ER Head   → P(ER+)
       │
       ├── PR Head   → P(PR+)
       │
       └── HER2 Head → P(HER2+)
```

Each receptor head therefore has its own parameters and target.

The outputs may be represented as binary logits or probabilities depending on the implementation.

Using separate heads is preferable to a single mutually exclusive receptor classifier because ER, PR, and HER2 are independent biological receptor-status variables.

---

# Survival Prognosis Head

### Module

```text
src/models/survival/deepsurv_head.py
```

The survival branch uses a DeepSurv-style neural network based on the Cox proportional hazards formulation.

The fused representation is mapped to a scalar risk score:

```text
h_f → DeepSurv → r
```

where:

```text
r ∈ R
```

is the predicted log-risk score.

The Cox proportional hazards model defines the relative hazard as:

```text
h(t | x) = h₀(t) exp(r)
```

where:

* `h(t | x)` is the conditional hazard
* `h₀(t)` is the baseline hazard
* `r` is the neural network risk score

The survival training objective is based on the Cox partial likelihood.

The model does not directly require estimation of the baseline hazard during training to learn relative risk.

---

# Multi-Task Learning

The framework jointly optimizes the classification and survival objectives.

The total loss can be represented conceptually as:

```text
L_total =
    λ_pam50 L_pam50
  + λ_er    L_er
  + λ_pr    L_pr
  + λ_her2  L_her2
  + λ_surv  L_survival
```

where:

* `L_pam50` = PAM50 multi-class classification loss
* `L_er` = ER binary classification loss
* `L_pr` = PR binary classification loss
* `L_her2` = HER2 binary classification loss
* `L_survival` = Cox/DeepSurv loss
* `λ_*` = task-specific loss weights

The task weights are configuration parameters rather than hardcoded values.

This allows the optimization contribution of the classification and survival objectives to be controlled during experimentation.

---

# Training

Training is orchestrated by:

```text
src/training/trainer.py
```

The trainer is responsible for:

1. Loading the configured dataset.
2. Constructing training/validation/test loaders.
3. Executing forward passes.
4. Computing task-specific losses.
5. Combining losses using configured task weights.
6. Performing backpropagation.
7. Updating model parameters.
8. Running validation.
9. Saving checkpoints.
10. Applying early stopping where configured.
11. Logging training and validation metrics.

Training configuration is stored under:

```text
configs/brca/train.yaml
```

The training pipeline should support:

* Mixed precision where hardware permits.
* Gradient clipping.
* Gradient accumulation where required.
* Configurable batch size.
* Learning-rate scheduling.
* Early stopping.
* Checkpointing.
* Reproducibility controls.

Because the framework is developed under constrained computational resources, the architecture should prioritize **performance per parameter** and efficient training rather than unnecessarily large model capacity.

---

# Evaluation

Evaluation is handled by:

```text
src/evaluation/evaluator.py
```

Metrics are implemented under:

```text
src/evaluation/metrics.py
```

Evaluation is performed independently for each prediction task.

## PAM50 Classification Metrics

Relevant metrics include:

* Accuracy
* Macro-F1
* Per-class precision
* Per-class recall
* AUROC
* Confusion matrix

Because PAM50 is a five-class problem and class imbalance may exist, macro-averaged metrics should receive particular attention.

## Receptor Classification Metrics

For ER, PR, and HER2:

* Accuracy
* Precision
* Recall
* F1-score
* AUROC
* Confusion matrix

Where class imbalance exists, AUROC and macro/balanced metrics should be considered alongside accuracy.

## Survival Metrics

The survival branch is evaluated using:

* Concordance index (C-index)
* Time-dependent AUROC where implemented
* Brier score where implemented

The C-index evaluates the model's ability to correctly rank patients according to relative survival risk.

---

# Interpretability

Interpretability should be designed around the actual modalities used by the model.

SHAP attribution over a joint genomic + clinical input space has been implemented
for the frozen checkpoint in `scripts/run_shap.py`; see
[experiments.md](experiments.md) for what is explained, the sanity checks
performed, and the interpretation limits.

Potential methods include:

### Genomic Interpretability

* Integrated Gradients
* Feature attribution
* Attention-based analysis

These methods can be used to identify genes contributing strongly to model predictions.

### Clinical Interpretability

* SHAP
* Integrated Gradients where appropriate
* Feature permutation/importance analysis

These methods can be used to investigate the contribution of clinical variables.

### Fusion Interpretability

Attention weights can be analyzed to investigate learned interactions between genomic and clinical representations.

Interpretability results should be treated as model explanations rather than direct evidence of biological causality.

---

# Inference

The primary inference entry point is:

```text
src/inference/predict.py
```

Inference should use the same preprocessing and feature-construction logic used during training.

Conceptually:

```text
Patient Input
     │
     ↓
Validation
     │
     ↓
Preprocessing
     │
     ├───────────────┐
     ↓               ↓
Gene Expression   Clinical Data
     │               │
     ↓               ↓
Genomic Transformer  Clinical MLP
     │               │
     └───────┬───────┘
             ↓
    Cross-Modal Attention
             │
             ↓
       Fusion MLP
             │
             ↓
     Fused Representation
             │
      ┌──────┼──────┐
      ↓      ↓      ↓
    PAM50  Receptors Survival
```

Inference outputs should contain structured results for all enabled tasks, including:

* PAM50 predicted class and probabilities.
* ER predicted status/probability.
* PR predicted status/probability.
* HER2 predicted status/probability.
* Survival risk score.

The inference path should support CPU execution.

---

# Configuration

The architecture is configuration-driven.

A BRCA-specific configuration structure should be organized approximately as:

```text
configs/
└── brca/
    ├── data.yaml
    ├── model.yaml
    └── train.yaml
```

The configuration should control:

### Data

* Dataset locations
* Gene-expression feature definitions
* PAM50 feature set
* Clinical feature definitions
* Preprocessing parameters
* Train/validation/test split configuration

### Model

* Genomic Transformer dimensions
* Number of Transformer layers
* Number of attention heads
* Clinical MLP dimensions
* Cross-modal attention dimensions
* Fusion MLP dimensions
* Dropout
* Prediction-head configuration

### Training

* Batch size
* Learning rate
* Optimizer
* Scheduler
* Number of epochs
* Loss weights
* Early stopping
* Gradient clipping
* Mixed precision

Architecture-specific parameters should not be scattered throughout source code.

---

# Project Structure

The architecture maps to the repository approximately as follows:

```text
src/
├── data/
│   ├── schema/
│   ├── preprocessing/
│   └── datasets/
│
├── models/
│   ├── genomics/
│   │   └── genomic_transformer.py
│   │
│   ├── tabular/
│   │   └── clinical_mlp.py
│   │
│   ├── fusion/
│   │   ├── cross_modal_attention.py
│   │   └── fusion_head.py
│   │
│   ├── classification/
│   │   ├── pam50_head.py
│   │   └── receptor_head.py
│   │
│   ├── survival/
│   │   └── deepsurv_head.py
│   │
│   └── model_factory.py
│
├── training/
│   ├── trainer.py
│   ├── losses.py
│   └── callbacks.py
│
├── evaluation/
│   ├── evaluator.py
│   ├── metrics.py
│   └── interpretability.py
│
└── inference/
    └── predict.py
```

The exact filenames may differ from the implementation, but the architectural separation should remain consistent.

---

# Design Principles

## Modularity

Each major model component should be independently constructible and testable:

* Genomic Transformer
* Clinical MLP
* Cross-Modal Attention
* Fusion MLP
* PAM50 head
* Receptor heads
* DeepSurv head

The model factory should compose these components without embedding task-specific implementation throughout the codebase.

## Multimodal Representation Learning

The model should learn representations from both genomic and clinical modalities rather than relying exclusively on either modality.

The architecture explicitly separates:

```text
Modality Encoding
       ↓
Cross-Modal Interaction
       ↓
Representation Fusion
       ↓
Multi-Task Prediction
```

## Multi-Task Learning

The classification and survival tasks share the learned multimodal representation while maintaining independent task-specific prediction heads.

This allows shared information to be learned from related clinical and molecular objectives while preserving separate output spaces.

## Config-Driven Design

Dataset paths, feature definitions, model dimensions, optimization parameters, and task weights should be controlled through configuration files.

## Data Leakage Prevention

All preprocessing operations that learn parameters from the data must respect the training/validation/test boundary.

In particular:

* Imputation statistics must not be estimated from the test set.
* Normalization parameters must be fitted using training data.
* Feature-selection decisions must not use held-out test labels.
* Patient overlap between train/validation/test partitions must be prevented.

## Reproducibility

Experiments should record:

* Configuration files
* Random seeds
* Dataset split information
* Model checkpoint
* Training metrics
* Evaluation metrics
* Software/environment information

This allows experiments to be reproduced and compared systematically.

## Computational Efficiency

The system is designed to be trainable on constrained hardware.

Therefore, model capacity should be justified experimentally rather than increased arbitrarily.

Particular attention should be paid to:

* Parameter count
* Memory consumption
* Batch size
* Training time
* Inference latency

---

# Experimental Baselines

The multimodal architecture should be evaluated against appropriate unimodal and simplified baselines.

The genomics-only and clinical-only baselines below have been run. Their measured
results, the exact reproduction commands, and the caveats that must accompany the
numbers are in [experiments.md](experiments.md). The simple-concatenation
baseline has not been run.

Recommended baselines include:

### Genomics-only

```text
Gene Expression
      ↓
Genomic Transformer
      ↓
Prediction Heads
```

### Clinical-only

```text
Clinical Data
      ↓
Clinical MLP
      ↓
Prediction Heads
```

### Simple Concatenation

```text
Genomic Embedding ──┐
                    ├── Concatenation → MLP → Heads
Clinical Embedding ─┘
```

### Proposed Multimodal Architecture

```text
Genomic Embedding ──┐
                    ├── Cross-Modal Attention
Clinical Embedding ─┘
                           ↓
                     Concatenation
                           ↓
                       Fusion MLP
                           ↓
                    Multi-Task Heads
```

These comparisons are important because improved performance should be attributable to the multimodal and cross-modal design rather than simply increased parameter count.

---

# Extending the Architecture

Although the current implementation is specifically designed for TCGA-BRCA, individual components remain modular.

## Adding a New Prediction Task

A new prediction task should generally require:

1. A new target in the dataset/schema layer.
2. A task-specific prediction head.
3. A corresponding loss function.
4. A configured task weight.
5. Evaluation metrics.
6. Optional inference output.

The genomic encoder, clinical encoder, cross-modal attention, and fusion layers should not require modification unless the new task requires a fundamentally different representation.

## Adding a New Clinical Feature

A new clinical variable should be added through the configuration and preprocessing pipeline rather than hardcoded into the model.

The Clinical MLP input dimension should be derived from the configured feature representation.

## Adding Another Genomic Feature Set

Alternative genomic feature sets can be evaluated by changing the feature-selection configuration and ensuring that the resulting input dimensionality is compatible with the Genomic Transformer.

The PAM50 feature set remains the primary genomic representation for the molecular subtype objective in the current experimental design.

---

# Current Architecture Summary

The final proposed architecture is:

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
                          ┌─────────────┴─────────────┐
                          │                           │
                          ↓                           ↓
         Attended Genomic Representation   Attended Clinical Representation        
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
                        Fused Multimodal Representation
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ↓                   ↓                   ↓
              PAM50 Head         Receptor Heads        DeepSurv Head
                    │              ┌────┼────┐                │
                    ↓              ↓    ↓    ↓                ↓
              5-Class             ER   PR   HER2         Risk Score
            Classification      Status Status Status
```

In mathematical form, the architecture can be summarized as:

```text
h_g = GenomicTransformer(x_g)

h_c = ClinicalMLP(x_c)

(h'_g, h'_c) = CrossModalAttention(h_g, h_c)

h_f = FusionMLP([h'_g || h'_c])

ŷ_pam50 = PAM50Head(h_f)

ŷ_er    = ERHead(h_f)

ŷ_pr    = PRHead(h_f)

ŷ_her2  = HER2Head(h_f)

r       = DeepSurvHead(h_f)
```

The overall training objective is:

```text
L =
λ₁ L_PAM50
+ λ₂ L_ER
+ λ₃ L_PR
+ λ₄ L_HER2
+ λ₅ L_Survival
```

Thus, the system is best characterized as a **two-modality, intermediate-fusion, cross-modal-attention, multi-task deep learning architecture for TCGA-BRCA**.
               