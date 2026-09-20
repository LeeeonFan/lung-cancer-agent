# Foundation-Model Fusion for Lung Cancer Subtyping

## Headline result

| Outcome | Result |
|---|---|
| Selected fusion | Fixed weighted log-probability fusion |
| Fusion weights | Metadata 0.05 · UNI2 0.25 · Virchow2 0.05 · Prism2 0.65 |
| Test macro AUROC | **0.8351** (95% CI 0.7632–0.9007) |
| Test balanced accuracy | 0.3587 (95% CI 0.2338–0.4871) |
| Best single-model AUROC | UNI2: 0.8200 |
| Best single-model balanced accuracy | UNI2: 0.3817 |


| Data quirk | Policy |
|---|---|
| `8377886` occurs at ages 68 and 69 | Retain as two records; shared grouping ID keeps both in one split |
| Five patients have conflicting slide labels | Select numerically lowest available WSI and use its label |
| Missing images | Use lowest available WSI; exclude only when no image is available |

## Final cohort and split

| Class | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Acinar | 30 | 21 | 3 | 6 |
| Cribriform | 15 | 11 | 1 | 3 |
| In situ | 45 | 31 | 5 | 9 |
| Lepidic | 19 | 13 | 2 | 4 |
| Micropapillary | 26 | 18 | 3 | 5 |
| Papillary | 40 | 27 | 4 | 9 |
| Solid | 29 | 20 | 3 | 6 |
| **Total** | **204** | **141** | **21** | **42** |

| Split control | Value |
|---|---|
| Seed | 42 |
| Grouping IDs: train / validation / test | 141 / 21 / 41 |
| Stratification | 7-class subtype |

## Image processing and tiling

| Stage | Configuration / result |
|---|---|
| Selected raw images | 204 images included|
| Downsampling | 80× → 20×; scale 0.25; Lanczos3; JPEG Q95 |
| Downsample output | 25.27 GB; 39 minutes |
| Tissue mask | Saturation ≥0.08; value 0.20–0.97 |
| Morphology | 3×3 opening ×1; closing ×2 |
| Mask minimum component | 0.0005 |
| Tile size / stride | 224×224 / 224; no overlap |
| Minimum tissue fraction | 0.50 |
| Maximum tiles per slide | 1,024 |
| Selected tiles | 200,488 |
| Slides reaching cap | 184/204 |
| Tiles/slide: min / median / max | 169 / 1,024 / 1,024 |

## Foundation-model embeddings

| Model | Input | Tile representation | Slide aggregation | Final shape |
|---|---|---|---|---:|
| UNI2-h | 224×224 RGB; official transform | 1,536-d | Mean pooling | 204×1,536 |
| Virchow2 | 224×224 RGB; official transform | CLS 1,280 + mean patch 1,280 | Mean pooling | 204×2,560 |
| Prism2 | ≤1,024 Virchow2 CLS tokens | — | Pretrained slide encoder | 204×2,560 |
| Metadata | Age + sex | — | — | 204×2 |


## Original Locked Test Results

The original pipeline used `StandardScaler → LogisticRegression`. Hyperparameters were selected using repeated stratified 5-fold cross-validation with three repeats. The locked test set was evaluated once.

| Configuration | Test macro AUROC | 95% CI | Test balanced accuracy | 95% CI |
|---|---:|---:|---:|---:|
| Metadata | 0.7013 | 0.6077–0.7949 | **0.4214** | 0.2625–0.5784 |
| UNI2 | 0.8200 | 0.7400–0.8905 | 0.3817 | 0.2305–0.5639 |
| Virchow2 | 0.8106 | 0.7268–0.8811 | 0.3302 | 0.2109–0.4620 |
| Prism2 | 0.8079 | 0.7241–0.8893 | 0.3619 | 0.2411–0.4830 |
| Prism2 + metadata | 0.8079 | 0.7241–0.8893 | 0.3619 | 0.2411–0.4830 |
| **Fused log-probability** | **0.8351** | **0.7632–0.9007** | 0.3587 | 0.2338–0.4871 |

| Original-test conclusion | Result |
|---|---|
| Fusion achieved highest macro AUROC | **Yes** |
| Fusion achieved highest balanced accuracy among foundation models | No |
| Fusion strictly outperformed every foundation model on both metrics | **No** |
| Main failure | Zero recall for Cribriform and Lepidic |

## Epoch-Based Test Results

> Post-hoc development experiment using PyTorch Linear classifiers. Hyperparameters and epoch counts were selected using train-only repeated CV. The fixed validation set was evaluated once. The test set was not re-evaluated.

### Training protocol

| Item | Setting |
|---|---|
| Classifier | `StandardScaler → PyTorch Linear(input_dim, 7)` |
| Optimizer | AdamW |
| CV | Repeated stratified 5-fold × 3 repeats |
| CV splits per configuration | 15 |
| Tuning budget | 12 configurations per representation |
| Learning rates | 0.0001, 0.0003, 0.001 |
| Weight decay | 0.001, 0.01 |
| Class weighting | None, balanced |
| Maximum epochs | 100 |
| Early-stopping patience | 15 |
| Selection score | Mean composite − 0.5 × SEM |
| Test set | Not evaluated |

### Main epoch-based comparison

| Representation | Train-CV AUROC | Train-CV BA | Validation AUROC | Validation BA |
|---|---:|---:|---:|---:|
| Metadata | 0.5334 | 0.1516 | 0.5015 | 0.0857 |
| UNI2 | 0.8160 | 0.4788 | 0.7273 | 0.3286 |
| Virchow2 | 0.8099 | 0.4808 | 0.7870 | 0.3762 |
| **Prism2** | **0.8768** | 0.5585 | **0.9319** | **0.6500** |
| Prism2 + metadata | 0.8678 | **0.5700** | 0.9118 | 0.5786 |
| Early concatenation | 0.8664 | 0.5590 | 0.8607 | 0.4238 |
| Early L2 concatenation | 0.8648 | 0.5628 | 0.8623 | 0.4238 |

### Selected Linear configurations

| Representation | Learning rate | Weight decay | Class weight | Fixed epochs |
|---|---:|---:|---|---:|
| Metadata | 0.0003 | 0.001 | balanced | 1 |
| UNI2 | 0.0010 | 0.001 | balanced | 9 |
| Virchow2 | 0.0003 | 0.001 | balanced | 13 |
| Prism2 | 0.0010 | 0.010 | none | 6 |
| Prism2 + metadata | 0.0010 | 0.010 | balanced | 5 |
| Early concatenation | 0.0003 | 0.001 | balanced | 5 |
| Early L2 concatenation | 0.0003 | 0.001 | balanced | 5 |

### Late-fusion results

| Strategy | Weights: Metadata / UNI2 / Virchow2 / Prism2 | OOF AUROC | OOF BA | Validation AUROC | Validation BA |
|---|---|---:|---:|---:|---:|
| Probability averaging | 0.05 / 0.20 / 0.10 / 0.65 | 0.8687 | **0.5237** | 0.9001 | **0.6857** |
| **Log-probability fusion** | **0.05 / 0.15 / 0.10 / 0.70** | **0.8741** | 0.5226 | **0.9077** | **0.6857** |
| Prism2 alone | — | 0.8613 | 0.5017 | **0.9319** | 0.6500 |

### Selected epoch-based finalist

| Item | Result |
|---|---|
| Selected strategy | **Log-probability fusion** |
| Fusion weights | Metadata 0.05 · UNI2 0.15 · Virchow2 0.10 · Prism2 0.70 |
| Validation macro AUROC | 0.9077 |
| Validation balanced accuracy | **0.6857** |
| Best single-model validation AUROC | Prism2: **0.9319** |
| Best single-model validation BA | Prism2: 0.6500 |
| Fusion strictly beats Prism2 on both metrics | **No** |
| Test set evaluated | **No** |

### Training-curve artifacts

| Artifact | Location |
|---|---|
| Full 200-epoch Prism2 diagnostic | `artifacts/results/linear_training/prism2/training_curves.png` |
| CV-locked training curves | `artifacts/results/linear_validation/<representation>/training_curves.png` |
| Epoch histories | `artifacts/results/linear_validation/<representation>/training_history.csv` |