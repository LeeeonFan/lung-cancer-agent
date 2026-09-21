# Lung Cancer Foundation-Model Fusion

Fusion of UNI2-h, Virchow2, Prism2, and patient metadata for 7-class lung adenocarcinoma subtype prediction.

## Dataset and split

| Item | Value |
|---|---:|
| Metadata slides | 408 |
| Available repository images | 386 |
| Included patient records | 204 |
| Independent grouping IDs | 203 |
| Excluded patients without available images | 6 |
| Classes | 7 |
| Split seed | 42 |

| Split | Records | Grouping IDs |
|---|---:|---:|
| Train | 141 | 141 |
| Fixed validation | 21 | 21 |
| Test | 42 | 41 |

All splits were created at the `SampleNumber` grouping level. The two records sharing grouping ID `8377886` were assigned to the same split.

## Data preparation

| Stage | Result |
|---|---:|
| Selected images | 204 |
| Downsampling | 80× → 20× |
| Maximum tiles per slide | 1,024 |
| Selected tiles | 200,488 |
| Slides reaching tile cap | 184/204 |

One available slide was selected deterministically per patient. When the lowest-numbered slide was unavailable, the lowest available slide was used. Patients with no available image were excluded.

## Foundation-model embeddings

| Model | Role | Tile/slide aggregation | Slide dimension |
|---|---|---|---:|
| UNI2-h | Tile encoder | Mean pooling | 1,536 |
| Virchow2 | Tile encoder | CLS token + mean patch token, followed by slide mean pooling | 2,560 |
| Prism2 | Slide encoder | Native Prism2 slide encoder | 2,560 |

| Cached representation | Shape |
|---|---:|
| UNI2-h | 204 × 1,536 |
| Virchow2 | 204 × 2,560 |
| Prism2 | 204 × 2,560 |
| Metadata | 204 × 2 |


## Experimental sequence

| Stage | Experiment | Main result | Decision |
|---:|---|---|---|
| 1 | PyTorch Linear baselines | Strongest train-CV representation: Prism2 + metadata | Continue with PyTorch Linear |
| 2 | Small MLP vs Linear | MLP average selection-score change: −0.00118; MLP won 3/6 image representations | Keep Linear |
| 3 | Early fusion | L2 concatenation score: 0.70784 | Retain as early-fusion baseline |
| 4 | PCA early fusion | Score: 0.61952 | Reject |
| 5 | Probability/log fusion | Log fusion score: 0.69159 | Retain |
| 6 | Temperature calibration | Best score: 0.69227; gain over log fusion: +0.00068 | Insufficient gain |
| 7 | Class-specific fusion | Best score: 0.69174; gain: +0.00015 | Reject |
| 8 | Fixed validation | Global log fusion had best finalist composite | Select global log fusion |
| 9 | Locked PyTorch test | Fusion did not beat every single model on both metrics | Requirement not met |

## Classifier-family comparison

Both classifier families used the same:

- seven representations;
- 12 hyperparameter configurations;
- 5-fold × 3-repeat train-only CV;
- seeds, preprocessing, metrics, and stopping policy.

| Representation | Linear score | MLP score | MLP − Linear |
|---|---:|---:|---:|
| UNI2 | 0.64046 | 0.64143 | +0.00097 |
| Virchow2 | **0.63995** | 0.62889 | −0.01106 |
| Prism2 | 0.71211 | **0.71343** | +0.00132 |
| Prism2 + metadata | 0.71336 | **0.72074** | +0.00739 |
| Early concat | **0.70727** | 0.70678 | −0.00049 |
| Early L2 concat | **0.70784** | 0.70266 | −0.00518 |

**Selected downstream classifier: PyTorch Linear.**

## PyTorch Linear train-CV baselines

| Representation | Macro AUROC | Balanced accuracy | Selection score | Epochs |
|---|---:|---:|---:|---:|
| Metadata | 0.53337 | 0.15157 | 0.33358 | 1 |
| UNI2 | 0.81599 | 0.47884 | 0.64046 | 9 |
| Virchow2 | 0.80987 | 0.48084 | 0.63995 | 13 |
| Prism2 | **0.87684** | 0.55855 | 0.71211 | 6 |
| Prism2 + metadata | 0.86783 | **0.57000** | **0.71336** | 5 |
| Early concat | 0.86635 | 0.55896 | 0.70727 | 5 |
| Early L2 concat | 0.86484 | 0.56277 | 0.70784 | 5 |

Selection score:

\[
\text{mean}\left(
\frac{\text{macro AUROC}+\text{balanced accuracy}}{2}
\right)
-0.5\times\text{SEM}
\]

## Fusion search

### PyTorch Linear OOF streams

| Stream | Macro AUROC | Balanced accuracy | Composite |
|---|---:|---:|---:|
| Metadata | 0.50248 | 0.17500 | 0.33874 |
| UNI2 | 0.80281 | 0.40081 | 0.60181 |
| Virchow2 | 0.79461 | 0.40541 | 0.60001 |
| Prism2 | **0.86132** | **0.50170** | **0.68151** |

### Late fusion

| Method | Macro AUROC | Balanced accuracy | Selection score |
|---|---:|---:|---:|
| Probability fusion | 0.86871 | **0.52372** | 0.69008 |
| Log-probability fusion | **0.87409** | 0.52261 | **0.69159** |
| Calibrated probability fusion | 0.87098 | **0.52610** | 0.69227 |
| Class-specific fusion | 0.87438 | 0.52261 | 0.69174 |

Calibration and class-specific fusion did not exceed the predeclared minimum material improvement of `0.001` over the existing log-fusion incumbent.

### Locked fusion

| Stream | Weight |
|---|---:|
| Metadata | 0.05 |
| UNI2 | 0.15 |
| Virchow2 | 0.10 |
| Prism2 | 0.70 |

## Fixed-validation finalists

| Strategy | Macro AUROC | Balanced accuracy | Composite |
|---|---:|---:|---:|
| Prism2 | **0.93187** | 0.65000 | 0.79094 |
| Probability fusion | 0.90008 | **0.68571** | 0.79290 |
| Log-probability fusion | 0.90772 | **0.68571** | **0.79672** |

**Selected fusion: global weighted log-probability fusion.**

## Final PyTorch test results

| Configuration | Macro AUROC (95% CI) | Balanced accuracy (95% CI) |
|---|---:|---:|
| Metadata | 0.5091 [0.3958, 0.6262] | 0.2389 [0.1403, 0.3571] |
| UNI2 | 0.7720 [0.6726, 0.8582] | **0.4770** [0.3430, 0.6190] |
| Virchow2 | 0.7283 [0.6313, 0.8153] | 0.3063 [0.1967, 0.4161] |
| Prism2 | **0.8011** [0.7114, 0.8898] | 0.3222 [0.2007, 0.4433] |
| Prism2 + metadata | 0.7978 [0.7122, 0.8845] | 0.3587 [0.2293, 0.4881] |
| Fused log probability | 0.7920 [0.7068, 0.8676] | 0.3381 [0.2242, 0.4578] |

Bootstrap confidence intervals used 2,000 grouping-ID cluster resamples.

**Fusion did not strictly outperform every single-foundation-model baseline on both primary metrics.**

## Test failure analysis

| Class | UNI2 recall | Prism2 recall | Fusion recall | Fusion AUROC |
|---|---:|---:|---:|---:|
| Acinar | **0.500** | 0.333 | 0.333 | 0.634 |
| Cribriform | 0.000 | 0.000 | 0.000 | 0.701 |
| In situ | **0.778** | 0.667 | **0.778** | 0.933 |
| Lepidic | **0.750** | 0.000 | 0.000 | 0.678 |
| Micropapillary | 0.200 | 0.200 | 0.200 | 0.870 |
| Papillary | 0.444 | **0.556** | **0.556** | 0.848 |
| Solid | **0.667** | 0.500 | 0.500 | 0.880 |

The largest balanced-accuracy deficit came from Lepidic: UNI2 correctly classified 3/4 cases, whereas Prism2 and the fused model classified 0/4. Cribriform recall was zero for all three models. Nonzero Lepidic and Cribriform AUROCs indicate that ranking signal remained, but those classes failed at the final argmax decision stage.

## Previous sklearn pipeline

| Pipeline | Fusion macro AUROC | Fusion balanced accuracy |
|---|---:|---:|
| sklearn Logistic Regression | **0.8351** | **0.3587** |
| PyTorch Linear | 0.7920 | 0.3381 |

The sklearn result is retained as the original locked test evaluation. The PyTorch result is a second-stage classifier migration using the same patient split.

## Training curves

Training curves include training loss, validation loss, validation macro AUROC, and validation balanced accuracy.

| Representation | Plot |
|---|---|
| Metadata | [Training curves](docs/figures/training/metadata.png) |
| UNI2 | [Training curves](docs/figures/training/uni2.png) |
| Virchow2 | [Training curves](docs/figures/training/virchow2.png) |
| Prism2 | [Training curves](docs/figures/training/prism2.png) |
| Prism2 + metadata | [Training curves](docs/figures/training/prism2_metadata.png) |
| Early concat | [Training curves](docs/figures/training/early_concat.png) |
| Early L2 concat | [Training curves](docs/figures/training/early_l2_concat.png) |

## Main commands

```bash
uv sync

uv run python scripts/build_patient_manifest.py
uv run python scripts/make_splits.py
uv run python scripts/build_slide_embeddings.py
uv run python scripts/validate_slide_embeddings.py

uv run python scripts/run_linear_cv.py --representation prism2
uv run python scripts/run_mlp_cv.py --representation prism2
uv run python scripts/compare_classifier_families.py

uv run python scripts/run_pca_linear_cv.py
uv run python scripts/run_temperature_calibration.py
uv run python scripts/run_class_specific_fusion.py

uv run python scripts/run_pytorch_final_test.py
```

## Verification

```bash
uv run ruff check .
uv run pytest
```

Current automated-test status: **15 passed**.