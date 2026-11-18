# Carotid Plaque Classifier

This repository contains three related components for carotid plaque research:

1. A PyTorch framework for **binary plaque-stability classification** (stable
   `0` vs. unstable `1`) from volumetric ultrasound/imaging sequences.
2. An independent Transformer model for standardized numerical tabular features.
3. R scripts for outpatient ultrasound ridge regression and survival analysis.

The image-classification framework uses a 3D ResNet whose slice-level features
can be enhanced by a **layer-relation attention module**, including an
L2-normalized scaled-dot-product variant. Training follows a strict 10-fold
stratified cross-validation protocol with a per-fold validation set for
checkpoint selection and a separate per-fold test set for reporting.

---

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [Preprocessing](#preprocessing)
- [Installation](#installation)
- [Usage](#usage)
- [Available Models](#available-models)
- [Tabular Feature Model](#tabular-feature-model)
- [Ultrasound Ridge and Survival Analysis](#ultrasound-ridge-and-survival-analysis)
- [Outputs](#outputs)
- [Evaluation Metrics](#evaluation-metrics)

---

## Key Features

- **3D ResNet backbone** (`BasicBlock3D`) with `GroupNorm` (no running-statistics
  batch normalization, so evaluation is deterministic per-sample).
- **Layer-relation attention** (`LayerRelationModule`) that models inter-slice
  relations in the channel dimension.
- **L2-normalized scaling variant** (`LayerRelationModuleScalingL2`) with
  query/key L2 normalization and `1/sqrt(d)` temperature scaling.
- **2D baseline / ablation models** (per-slice ResNet2D with mean aggregation,
  optionally with a slice-relation attention module).
- **10-fold stratified cross-validation** with train / validation / test split.
- **Class-imbalance handling**: balanced random sampling and optional
  loss class-weights computed from the *training fold only*.
- **Attention artifact export** for qualitative interpretation.
- **Automatic logging**: per-fold best models, loss curves, CSV loss logs, and
  a JSON summary of all metrics.
- **Numerical tabular Transformer** with feature-specific tokenization and a
  learnable classification token.
- **Outpatient statistical analysis** with IPCW-weighted ridge models,
  time-to-event analyses, sensitivity analyses, and Kaplan-Meier plots.

---

## Project Structure

```
CarotidPlaqueClassifier/
├── train.py                              # Main training entry point (10-fold CV)
├── carotid_dataloader.py                 # Dataset + DataLoader builders
├── models/
│   ├── __init__.py                       # Model registry + build_model()
│   ├── classifier_head.py                # Shared MLP classification head
│   ├── resnet3d_attention.py             # ResNet3D + LayerRelationModule
│   ├── resnet3d_attention_scaling_l2.py  # L2-normalized scaling attention variant
│   └── ablation_models.py                # ResNet3D / ResNet2D / ResNet2DAttention
├── feature/
│   └── models/
│       └── tabular_transformer.py         # Transformer for numerical tabular features
├── Ultrasound_Ridge_and_Survival_Code/
│   ├── README.md                          # Detailed R workflow and input schema
│   ├── reanalyze_outpatients_current.R    # Primary ridge and survival analysis
│   ├── outpatient_sensitivity_current.R   # Sensitivity and supplementary analyses
│   └── outpatient_km_ci_current.R          # Kaplan-Meier plots with confidence intervals
├── utils/
│   ├── io.py                             # Experiment dir, config/history/log helpers
│   ├── metrics.py                        # Classification metrics
│   ├── plotting.py                       # Loss-curve plotting
│   ├── training.py                       # Seed, train/eval loops
│   └── attention_export.py               # Attention artifact export
└── preprocess/
    ├── crop_image_by_mask.py             # Crop carotid ROI using segmentation masks
    └── clean_tabular_features.py         # Clean clinical tabular features
```

---

## Data Format

The dataset is loaded by `CarotidPlaqueDataset` (`carotid_dataloader.py`).

- **`--image_root`**: a directory containing **one subdirectory per sample**.
  Each subdirectory is named `{patient}_{side}` (e.g. `ZhangSan_left`) and holds
  a sequence of grayscale slices (`.png`, `.jpg`, `.jpeg`). Slices are sorted in
  natural order and the middle `n_images` slices are selected.
- **`--excel_path`**: an `.xlsx` workbook with at least two columns:
  - `name`: matches the subdirectory names.
  - `label`: integer label, `0` = stable, `1` = unstable.

Images are loaded in grayscale (`image_mode="L"`) and normalized to `[0, 1]` by
dividing by 255.

---

## Preprocessing

Two standalone preprocessing scripts are provided.

### 1. Crop carotid ROI from masks

```bash
python preprocess/crop_image_by_mask.py
```

Crops the 72×72 carotid region from each original image using segmentation
masks. The `left`/`right` suffix of the sample name is used for a mirrored
box-selection heuristic. Paths are hard-coded at the top of the script.

### 2. Clean clinical tabular features

```bash
python preprocess/clean_tabular_features.py \
    --input feature/data.xlsx \
    --sheet-name "合并数据"
```

Produces three workbooks:

- `*_standardized.xlsx` — every row/column retained, formatting errors fixed,
  genuine missing values left for **train-only imputation**.
- `*_complete.xlsx` — a zero-missing subset obtained by greedily removing
  rows/columns (trade-off controlled by `--row-priority`).
- `*_cleaning_report.xlsx` — corrections, removed rows/columns, missing
  statistics, and suspicious values.

---

## Installation

Python 3.10+ is recommended.

```bash
pip install torch numpy pandas openpyxl pillow scikit-learn matplotlib pyyaml opencv-python
```

The outpatient analysis is separate from the Python workflow. Its original
environment used R 4.3.3 with `survival`, `glmnet`, `Matrix`, `ggplot2`, and
`ragg`; see its [dedicated README](Ultrasound_Ridge_and_Survival_Code/README.md)
for the recorded package versions.

---

## Usage

Run 10-fold cross-validation training:

```bash
python train.py \
    --image_root /media/data/wjf/data/Carotid_artery_crop_old \
    --excel_path /media/data/wjf/data/Carotid_artery_excel.xlsx \
    --model_name resnet3d_attention_scalingL2 \
    --output_root runs \
    --epochs 30 \
    --batch_size 8 \
    --num_folds 10
```

Export attention artifacts for the test set of each fold:

```bash
python train.py \
    --model_name resnet3d_attention_scalingL2 \
    --export_attention_artifacts \
    --attention_export_batches 1
```

---

## Available Models

Pass one of the following to `--model_name`:

| `--model_name` | Description |
| --- | --- |
| `resnet3d` | Plain 3D ResNet backbone |
| `resnet3d_attention` | 3D ResNet + layer-relation attention |
| `resnet3d_attention_scalingL2` | 3D ResNet + L2-normalized scaling attention |
| `resnet2d` | Per-slice 2D ResNet with mean aggregation (ablation) |
| `resnet2d_attention` | 2D ResNet + slice-relation attention (ablation) |

---

## Tabular Feature Model

[`feature/models/tabular_transformer.py`](feature/models/tabular_transformer.py)
provides `TabularTransformerBinaryClassifier`, an independent binary classifier
for standardized numerical features. It is not part of the image-model registry
or the `train.py` workflow.

For an input tensor shaped `[batch_size, num_features]`, the model:

1. Converts every scalar into a feature-specific token using a learned weight
   and bias.
2. Prepends a learnable `[CLS]` token.
3. Processes the token sequence with a pre-normalized Transformer encoder using
   GELU activations.
4. Sends the encoded `[CLS]` representation through a LayerNorm/MLP head and
   returns one logit per sample with shape `[batch_size]`.

```python
import torch

from feature.models.tabular_transformer import TabularTransformerBinaryClassifier

model = TabularTransformerBinaryClassifier(
    num_features=12,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.2,
)

standardized_features = torch.randn(8, 12)
logits = model(standardized_features)  # shape: [8]
probabilities = logits.sigmoid()
```

`d_model` must be divisible by `nhead`. Since the model returns raw logits,
`torch.nn.BCEWithLogitsLoss` is the appropriate training loss for binary labels.

---

## Ultrasound Ridge and Survival Analysis

[`Ultrasound_Ridge_and_Survival_Code/`](Ultrasound_Ridge_and_Survival_Code/)
contains a separate R workflow for a 2,618-patient outpatient cohort. It uses a
frozen CTA probability as a baseline input and does **not** train the image
network in this repository.

Prepare the approved, analysis-ready data as `analysis_data_private.csv` in a
private working directory outside the Git checkout, then run:

```bash
Rscript Ultrasound_Ridge_and_Survival_Code/reanalyze_outpatients_current.R /path/to/private_run
Rscript Ultrasound_Ridge_and_Survival_Code/outpatient_sensitivity_current.R /path/to/private_run
Rscript Ultrasound_Ridge_and_Survival_Code/outpatient_km_ci_current.R /path/to/private_run
```

The primary script fits IPCW-weighted ridge logistic models at 12, 24, and 36
months with grouped nested cross-validation and produces discrimination,
calibration, decision-curve, Kaplan-Meier, and Cox analyses. The remaining
scripts add sensitivity analyses and confidence-interval KM figures. Exact input
fields, run-order details, and privacy requirements are documented in the
[ultrasound analysis README](Ultrasound_Ridge_and_Survival_Code/README.md).

No patient data, patient-level predictions, or intermediate analysis objects
are included in this repository.

---

## Outputs

Each run creates a timestamped directory under `--output_root`:

```
runs/<YYYYmmdd_HHMMSS>/
├── config.yaml                    # Full configuration snapshot
├── train.log                      # Per-fold training/evaluation log
├── best_model_fold{1..10}.pth     # Best checkpoint (selected on validation)
├── losses_10fold.csv              # Per-fold train/val loss curves
├── history.json                   # All fold results + metrics
├── loss_curves_10fold/            # Per-fold loss-curve PNGs
└── attention_artifacts/           # (optional) relation matrices + features
    └── fold_{id}/batch_XXX_<name>/
```

---

## Evaluation Metrics

Reported per fold for the test set (also tracked on validation):

- Accuracy, AUC (ROC), recall (sensitivity), precision, F1, specificity,
  balanced accuracy, NPV, Brier score, Cohen's kappa, and PR-AUC.

AUC / PR-AUC are set to `NaN` when a fold contains only a single class.

---
