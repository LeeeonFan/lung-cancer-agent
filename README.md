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
| Group leakage | None detected |
| Test use | Once, after finalist lock |

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


## Classifier and OOF protocol

| Item | Configuration |
|---|---|
| Classifier | `StandardScaler → multinomial LogisticRegression` |
| `C` grid | 0.0001, 0.001, 0.01, 0.1, 1, 10 |
| Class weights | None, balanced |
| Budget | 12 configurations per representation |
| CV | Repeated stratified 5-fold ×3 repeats = 15 splits |
| OOF predictions | 3 per patient per model |
| Composite | (macro AUROC + balanced accuracy) / 2 |
| Selection score | Mean composite − 0.5×SEM |

| Representation | Selected C | Class weight |
|---|---:|---|
| Metadata | 10 | balanced |
| UNI2-h | 0.01 | balanced |
| Virchow2 | 0.01 | none |
| Prism2 | 0.001 | none |
| Prism2 + metadata | 0.001 | none |

## Baselines

| Representation | Train-CV AUROC | Train-CV BA | Selection score | Validation AUROC | Validation BA |
|---|---:|---:|---:|---:|---:|
| Metadata | 0.5979 | 0.2516 | 0.4156 | 0.5594 | 0.1643 |
| UNI2-h | 0.8270 | 0.4388 | 0.6284 | 0.8346 | 0.4714 |
| Virchow2 | 0.8069 | 0.3975 | 0.5944 | 0.8537 | 0.4119 |
| Prism2 | 0.8691 | 0.5044 | 0.6780 | **0.9166** | **0.5429** |
| Prism2 + metadata | **0.8692** | **0.5044** | **0.6781** | **0.9166** | **0.5429** |

## Fusion search

| Strategy | Configuration | Macro AUROC | Balanced accuracy | Selection score |
|---|---|---:|---:|---:|
| Early concatenation | Raw 6,658-d | 0.8627 | 0.4976 | 0.6739 |
| Early L2 concatenation | Per-stream normalization | 0.8625 | 0.5035 | 0.6765 |
| Probability averaging | Weights .05/.20/.05/.70 | 0.8745 | 0.5138 | 0.6865 |
| Fixed log-probability | Weights .05/.25/.05/.65 | **0.8758** | 0.5189 | 0.6895 |
| Nested stacking | 28 probability features | 0.8436 | 0.4560 | 0.6424 |
| Metadata-gated fusion | Patient-specific weights | 0.8758 | **0.5208** | **0.6901** |


## Finalist validation

| Strategy | Macro AUROC | Balanced accuracy | Composite |
|---|---:|---:|---:|
| **Fixed log-probability** | **0.9062** | 0.5429 | **0.7245** |
| Metadata-gated | 0.8998 | 0.5429 | 0.7213 |



## Held-out test results

| Configuration | Macro AUROC | 95% CI | Balanced accuracy | 95% CI |
|---|---:|---:|---:|---:|
| Metadata | 0.7013 | 0.6077–0.7949 | **0.4214** | 0.2625–0.5784 |
| UNI2-h | 0.8200 | 0.7400–0.8905 | 0.3817 | 0.2305–0.5639 |
| Virchow2 | 0.8106 | 0.7268–0.8811 | 0.3302 | 0.2109–0.4620 |
| Prism2 | 0.8079 | 0.7241–0.8893 | 0.3619 | 0.2411–0.4830 |
| Prism2 + metadata | 0.8079 | 0.7241–0.8893 | 0.3619 | 0.2411–0.4830 |
| **Fused fixed log-probability** | **0.8351** | **0.7632–0.9007** | 0.3587 | 0.2338–0.4871 |

## Paired bootstrap: fusion minus comparator

| Comparator | AUROC Δ | 95% CI | P(fusion >) | BA Δ | 95% CI | P(fusion >) |
|---|---:|---:|---:|---:|---:|---:|
| Metadata | +0.1358 | +0.0071 to +0.2541 | 0.981 | −0.0583 | −0.2604 to +0.1424 | 0.280 |
| UNI2-h | +0.0156 | −0.0272 to +0.0607 | 0.754 | −0.0268 | −0.2001 to +0.1304 | 0.387 |
| Virchow2 | +0.0253 | −0.0275 to +0.0802 | 0.842 | +0.0295 | −0.1000 to +0.1476 | 0.669 |
| Prism2 | +0.0269 | −0.0125 to +0.0629 | 0.925 | −0.0040 | −0.0714 to +0.0714 | 0.411 |

## Per-class test AUROC

| Class | UNI2 | Virchow2 | Prism2 | Fused |
|---|---:|---:|---:|---:|
| Acinar | 0.8148 | **0.8380** | 0.6852 | 0.7546 |
| Cribriform | **0.8803** | 0.7863 | 0.7521 | 0.8120 |
| In situ | 0.9024 | 0.8956 | **0.9428** | **0.9428** |
| Lepidic | 0.7434 | 0.7039 | 0.7039 | **0.7697** |
| Micropapillary | 0.7730 | 0.7784 | **0.8378** | 0.8270 |
| Papillary | 0.7374 | 0.8013 | **0.8586** | 0.8552 |
| Solid | **0.8889** | 0.8704 | 0.8750 | 0.8843 |

## Fused confusion summary

| True class | Correct / total | Recall | Main errors |
|---|---:|---:|---|
| Acinar | 2/6 | 0.333 | Solid 2; Cribriform 1; Papillary 1 |
| Cribriform | 0/3 | 0.000 | Micropapillary 2; Solid 1 |
| In situ | 5/9 | 0.556 | Papillary 2; Lepidic 1; Micropapillary 1 |
| Lepidic | 0/4 | 0.000 | Papillary 3; In situ 1 |
| Micropapillary | 2/5 | 0.400 | Papillary 2; Solid 1 |
| Papillary | 5/9 | 0.556 | Acinar 2; In situ 2 |
| Solid | 4/6 | 0.667 | Cribriform 1; Micropapillary 1 |
