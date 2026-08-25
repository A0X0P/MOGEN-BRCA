# Data Sources

## Overview

This project uses the **The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA)** cohort as the primary data source for breast cancer molecular subtype classification, receptor-status classification, and survival prognosis.

The current framework uses **two data modalities only**:

1. **Gene expression**
2. **Clinical data**

Imaging data is **not part of the current implementation**.

The framework is designed around a patient-level matched cohort in which genomic, clinical, molecular-label, and survival information are aligned using TCGA patient identifiers.

---

# TCGA-BRCA

## Dataset

**TCGA-BRCA — Breast Invasive Carcinoma**

TCGA-BRCA provides molecular and clinical information from breast cancer patients and is the sole cancer cohort used by the current implementation.

The current dataset contains:

* Gene expression data
* Clinical/pathological data
* PAM50 molecular subtype labels where available
* ER receptor status
* PR receptor status
* HER2 receptor status
* Overall survival time
* Overall survival event/censoring information

The current modeling pipeline uses these data to construct a multimodal patient representation and perform multiple prediction tasks.

---

# Data Modalities

## 1. Gene Expression

### Source

The genomic modality is derived from the TCGA-BRCA RNA-seq expression matrix.

The current source contains approximately:

* **20,531 genes**
* **1,084 patient/sample columns** in the principal TCGA-BRCA PanCancer Atlas expression source
* **1,082 tumour samples** available for the final patient-level intersection used by the current data audit

The model does not use the complete 20,531-dimensional expression matrix directly.

Instead, the current architecture is designed around the **PAM50 gene-expression feature set**.

### Processing

The genomic preprocessing pipeline is responsible for:

* Selecting the required genomic features.
* Aligning gene identifiers.
* Ensuring a deterministic feature ordering.
* Handling invalid or missing values.
* Applying the configured expression transformation/normalization.
* Producing a fixed-dimensional genomic tensor for the Genomic Transformer.

The final genomic feature vocabulary must be explicitly defined and version-controlled.

**Important:** The repository must not invent or silently substitute a PAM50 gene list. If the authoritative PAM50 vocabulary is not present in the repository, training must remain blocked until the feature vocabulary is explicitly supplied and validated.

### Model Usage

The processed gene-expression vector is passed to:

```text
src/models/genomics/genomic_transformer.py
```

The resulting representation is the genomic embedding used by the multimodal fusion architecture.

---

# 2. Clinical Data

### Source

Clinical and pathological information is obtained from TCGA-BRCA clinical data sources.

The data audit identified two principal clinical sources:

```text
brca_tcga_clinical_data.tsv
brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt
```

The sources contain overlapping but not identical patient records and fields.

Patient identifiers must therefore be normalized before constructing the final cohort.

### Clinical Features

The current model configuration uses a **12-dimensional clinical representation**.

The clinical feature representation includes configured variables derived from clinical/pathological information, including:

* Age
* Tumour stage
* Nodal status
* Sex
* Encoded categorical components of the above variables where applicable

The exact encoded dimensionality is determined by the preprocessing/schema configuration rather than by the raw number of clinical columns.

The current model configuration specifies:

```text
clinical_input_dim = 12
```

### Processing

Clinical preprocessing is responsible for:

* Patient-ID normalization.
* Clinical variable selection.
* Missing-value handling.
* Categorical encoding.
* Numerical normalization/standardization where applicable.
* Consistent feature ordering.
* Conversion to tensors.

The resulting clinical tensor is passed to:

```text
src/models/tabular/clinical_mlp.py
```

---

# Molecular and Clinical Targets

The TCGA-BRCA data is used to construct four classification targets and one survival target.

## PAM50 Molecular Subtype

The molecular subtype target consists of five intrinsic molecular classes:

| Class         | Description                                                 |
| ------------- | ----------------------------------------------------------- |
| Luminal A     | Hormone-receptor-associated, generally lower proliferation  |
| Luminal B     | Hormone-receptor-associated, generally higher proliferation |
| HER2-enriched | HER2-associated intrinsic molecular subtype                 |
| Basal-like    | Basal-like intrinsic molecular subtype                      |
| Normal-like   | Normal-like intrinsic molecular subtype                     |

The model therefore performs:

```text
5-class PAM50 classification
```

The audited TCGA-BRCA cohort contained:

| PAM50 subtype | Count |
| ------------- | ----: |
| Luminal A     |   499 |
| Luminal B     |   197 |
| Basal-like    |   171 |
| HER2-enriched |    78 |
| Normal-like   |    36 |
| Missing       |   101 |
| Total         | 1,082 |

PAM50 labels were available for approximately **90.67%** of the audited cohort.

The missing-label cases are not treated as a sixth biological subtype.

---

# Receptor-Status Targets

The framework predicts three receptor-status targets independently.

## ER

Estrogen receptor status:

```text
ER-positive
ER-negative
```

The audited cohort contained:

* ER-positive: 794
* ER-negative: 237
* Missing/other: remaining cases

## PR

Progesterone receptor status:

```text
PR-positive
PR-negative
```

The audited cohort contained:

* PR-positive: 686
* PR-negative: 342
* Missing/other: remaining cases

## HER2

HER2 status requires explicit preprocessing because multiple measurement sources and equivocal results are present in the TCGA-BRCA clinical data.

The audited data showed:

* HER2 IHC-positive: 160
* HER2 IHC-negative: 556
* HER2 equivocal: 175
* HER2 missing: 179

FISH information can resolve a substantial proportion of equivocal cases.

The final HER2 labeling policy must therefore be explicitly configured and documented before the final training run.

The implementation must not silently treat an equivocal HER2 result as either positive or negative.

---

# Survival Data

Overall survival is derived from the TCGA-BRCA clinical data using:

```text
OS_MONTHS
OS_STATUS
```

These are the authoritative variables for the current survival objective.

The audited cohort contained:

* Total patients: **1,082**
* Living/censored: **931**
* Deceased/events: **151**
* Survival records with both required variables: **1,082**

Observed survival time ranged from:

```text
0 to approximately 282.9 months
```

with a median of approximately:

```text
27.07 months
```

The survival target is represented as:

```text
survival_time
event
```

and is passed to the DeepSurv/Cox survival objective.

---

# Cohort Construction

The final modeling cohort is constructed by intersecting patients available across the required data sources.

The read-only audit identified:

| Source                         | Available patients/records |
| ------------------------------ | -------------------------: |
| TCGA PanCancer clinical cohort |                      1,084 |
| Clinical TSV                   |                      1,101 |
| RNA-seq expression             |                      1,082 |
| Final three-way intersection   |                  **1,082** |

The current patient-level intersection therefore contains **1,082 patients with RNA-seq and core clinical information**.

Task-specific labels are available for different subsets of this cohort.

Consequently, the effective training population can differ by task depending on label availability.

---

# Clinical Data Availability

The audited availability of selected clinical variables is:

| Variable                 | Availability |
| ------------------------ | -----------: |
| Age                      |        1,082 |
| Tumour stage             |        1,063 |
| Nodal status             |        1,062 |
| ER                       |        1,031 |
| PR                       |        1,028 |
| HER2 IHC only            |          716 |
| HER2 IHC + FISH fallback |          942 |
| PAM50 subtype            |          981 |
| Overall survival         |        1,082 |

Missing values must be handled explicitly by the preprocessing pipeline and must not be converted into arbitrary biological labels.

---

# Patient Matching

All modalities and targets must be aligned at the **patient level**.

The preferred matching key is the normalized TCGA patient identifier.

The pipeline must ensure:

```text
Patient ID
    │
    ├── Gene Expression
    ├── Clinical Features
    ├── PAM50 Target
    ├── ER Target
    ├── PR Target
    ├── HER2 Target
    └── Survival Target
```

correspond to the same biological patient.

Sample-level identifiers must not be confused with patient-level identifiers when constructing the multimodal cohort.

Duplicate or ambiguous patient mappings must be detected during preprocessing rather than silently resolved.

---

# Data Leakage Considerations

The data pipeline must prevent information from the validation or test partitions from influencing training.

The following rules apply:

* Train/validation/test splits must be performed at the patient level.
* Patients must not appear in more than one partition.
* Normalization parameters must be estimated from training data only.
* Imputation parameters must be estimated from training data only.
* Feature-selection procedures must not use held-out test labels.
* Any learned preprocessing transformation must be fitted only on the training partition.

## PAM50 Label/Input Circularity

A specific issue exists for PAM50 classification.

PAM50 subtype labels are biologically derived from the expression of the PAM50 genes. Therefore:

```text
PAM50 genes → PAM50 classification target
```

creates a potential circularity if the model is evaluated as though it were predicting an independently measured outcome.

The current architecture must therefore document this explicitly.

The PAM50 classification experiment should be interpreted as **reproducing/classifying the PAM50 molecular subtype from its defining expression signature**, rather than as an independent clinical prediction task.

This issue must not be hidden in the methodology or evaluation.

---

# Demographic Variables

Race and ethnicity should not automatically be included in the primary predictive feature set.

Where demographic information is available, it can instead be used for:

* Cohort characterization
* Representation analysis
* Subgroup performance analysis
* Fairness/error analysis where sample sizes permit

This keeps demographic characterization conceptually separate from the primary molecular/clinical predictive representation.

---

# Files and Repository Locations

The current raw data is stored under the repository's raw-data area.

The principal sources identified during the audit include:

```text
data/raw/
├── brca_tcga_clinical_data.tsv
├── brca_tcga_pan_can_atlas_2018/
│   └── data_clinical_patient.txt
└── data_mrna_seq_v2_rsem.txt
```

The exact directory organization should remain consistent with the validated repository state.

No imaging directory is required for the current BRCA implementation.

---

# Data Processing Pipeline

The data flow is:

```text
TCGA-BRCA Raw Data
        │
        ↓
Patient-ID Normalization
        │
        ↓
Cohort Intersection
        │
        ├─────────────────────┐
        ↓                     ↓
Gene Expression         Clinical Data
        │                     │
        ↓                     ↓
PAM50 Feature Selection  Clinical Feature Selection
        │                     │
        ↓                     ↓
Genomic Preprocessing    Clinical Preprocessing
        │                     │
        ↓                     ↓
Genomic Tensor           Clinical Tensor
        │                     │
        └──────────┬──────────┘
                   ↓
             Model Dataset
```

Targets are constructed alongside the patient record:

```text
PAM50
ER
PR
HER2
OS time
OS event
```

---

# Data Access and Licensing

TCGA data is distributed through the National Cancer Institute's genomic data infrastructure.

The current project uses the publicly accessible data required for the defined gene-expression and clinical analyses.

The project does not require TCGA controlled-access raw sequencing data for the current modeling pipeline.

Data should be cited according to the appropriate TCGA/GDC publication and dataset requirements in the final research report.

The project should not introduce personally identifiable patient information into the repository.

TCGA patient identifiers are research dataset identifiers and must be treated as dataset keys rather than as personally identifying information for application purposes.

---

# Data Versioning and Reproducibility

The exact dataset/source version used for a training run should be recorded.

Each experiment should ideally preserve:

* Dataset source
* Dataset release/version where applicable
* Source file names
* Feature vocabulary version
* Clinical preprocessing configuration
* Cohort-selection rules
* Train/validation/test split
* Random seed
* Model configuration
* Training configuration
* Model checkpoint

The purpose is to ensure that a reported result can be reproduced from the same data and configuration.

---

# Current Data Status

The TCGA-BRCA source audit has established the core two-modality data foundation:

```text
                 TCGA-BRCA
                     │
          ┌──────────┴──────────┐
          │                     │
          ↓                     ↓
     Gene Expression       Clinical Data
          │                     │
          ↓                     ↓
       PAM50 genes        Clinical features
          │                     │
          └──────────┬──────────┘
                     │
                     ↓
              Multimodal Model
```

The following have been validated:

* TCGA-BRCA is the active cancer cohort.
* Gene expression data is available.
* Clinical data is available.
* A three-way patient intersection of 1,082 was identified.
* PAM50 labels are available for 981 patients.
* ER labels are available for 1,031 patients.
* PR labels are available for 1,028 patients.
* HER2 requires an explicit IHC/FISH labeling policy.
* Overall survival data is available for all 1,082 patients in the audited intersection.
* Age, tumour stage, and nodal status are available at high coverage.

---

# Blocking Data Decisions

The following decisions must be resolved before the final training run.

* [ ] **PAM50 gene vocabulary:** provide and validate the authoritative 50-gene feature list; do not invent a list.
* [ ] **HER2 labeling policy:** define the rule for IHC/FISH integration, equivocal results, and conflicting measurements.
* [ ] **PAM50 circularity:** explicitly document the relationship between PAM50 expression features and PAM50-derived labels in the experimental design.
* [ ] **Survival preprocessing:** finalize the explicit rule for zero/edge-case survival times.
* [ ] **Task-specific missing labels:** define how samples without a particular task label are masked during multi-task loss computation.
* [ ] **Final cohort manifest:** freeze and record the patient IDs used in each experiment after all inclusion/exclusion rules are applied.


---

# Authoritative Data Architecture

The current project data architecture is therefore:

```text
                         TCGA-BRCA
                             │
             ┌───────────────┴───────────────┐
             │                               │
             ↓                               ↓
      RNA-seq Gene Expression          Clinical Data
             │                               │
             ↓                               ↓
        PAM50 Features                 Clinical Features
             │                               │
             └───────────────┬───────────────┘
                             │
                             ↓
                    Matched Patient Cohort
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ↓              ↓              ↓
           PAM50          ER/PR/HER2      Survival
            Labels         Labels         OS_MONTHS/
                                         OS_STATUS
```

This document describes the **current TCGA-BRCA data source and cohort**, not a hypothetical future multi-cancer system.
