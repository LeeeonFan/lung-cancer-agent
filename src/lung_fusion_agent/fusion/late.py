from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from lung_fusion_agent.eval.metrics import (
    calculate_classification_metrics,
)


@dataclass(frozen=True)
class LateFusionResult:
    macro_auroc_mean: float
    macro_auroc_std: float
    balanced_accuracy_mean: float
    balanced_accuracy_std: float
    composite_mean: float
    composite_std: float
    composite_sem: float
    selection_score: float
    fold_metrics: pd.DataFrame


def validate_weights(
    weights: np.ndarray,
    *,
    expected_streams: int,
) -> np.ndarray:
    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    if weights.shape != (expected_streams,):
        raise ValueError(
            f"Expected weight shape ({expected_streams},), found {weights.shape}."
        )

    if not np.isfinite(weights).all():
        raise ValueError("Weights contain NaN or infinity.")

    if (weights < 0).any():
        raise ValueError("Late-fusion weights must be non-negative.")

    weight_sum = float(weights.sum())

    if weight_sum <= 0:
        raise ValueError("Late-fusion weights must have a positive sum.")

    return weights / weight_sum


def fuse_probabilities(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted arithmetic mean over model streams."""
    if probabilities.ndim != 3:
        raise ValueError("Probabilities must have shape (patients, streams, classes).")

    normalized_weights = validate_weights(
        weights,
        expected_streams=(probabilities.shape[1]),
    )

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinity.")

    fused = np.einsum(
        "nsc,s->nc",
        probabilities,
        normalized_weights,
    )

    row_sums = fused.sum(
        axis=1,
        keepdims=True,
    )

    if (row_sums <= 0).any():
        raise ValueError("Fused probability row has a non-positive sum.")

    fused = fused / row_sums

    return fused.astype(
        np.float32,
        copy=False,
    )


def evaluate_late_fusion(
    *,
    probability_cache: np.ndarray,
    validation_masks: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    weights: np.ndarray,
) -> LateFusionResult:
    if probability_cache.ndim != 4:
        raise ValueError(
            "Probability cache must have shape (folds, patients, streams, classes)."
        )

    if validation_masks.shape != (
        probability_cache.shape[0],
        probability_cache.shape[1],
    ):
        raise ValueError("Validation mask shape mismatch.")

    if labels.shape != (probability_cache.shape[1],):
        raise ValueError("Label shape mismatch.")

    normalized_weights = validate_weights(
        weights,
        expected_streams=(probability_cache.shape[2]),
    )

    fold_records = []

    for split_index in range(probability_cache.shape[0]):
        fold_mask = validation_masks[split_index]

        fold_probabilities = probability_cache[
            split_index,
            fold_mask,
            :,
            :,
        ]

        fused_probabilities = fuse_probabilities(
            fold_probabilities,
            normalized_weights,
        )

        metrics = calculate_classification_metrics(
            y_true=labels[fold_mask],
            probabilities=(fused_probabilities),
            class_names=class_names,
        )

        composite_score = (metrics.macro_auroc + metrics.balanced_accuracy) / 2.0

        fold_records.append(
            {
                "split_index": split_index,
                "macro_auroc": (metrics.macro_auroc),
                "balanced_accuracy": (metrics.balanced_accuracy),
                "composite_score": (composite_score),
            }
        )

    fold_metrics = pd.DataFrame(fold_records)

    composite_mean = float(fold_metrics["composite_score"].mean())

    composite_std = float(fold_metrics["composite_score"].std(ddof=1))

    composite_sem = composite_std / sqrt(len(fold_metrics))

    selection_score = composite_mean - 0.5 * composite_sem

    return LateFusionResult(
        macro_auroc_mean=float(fold_metrics["macro_auroc"].mean()),
        macro_auroc_std=float(fold_metrics["macro_auroc"].std(ddof=1)),
        balanced_accuracy_mean=float(fold_metrics["balanced_accuracy"].mean()),
        balanced_accuracy_std=float(fold_metrics["balanced_accuracy"].std(ddof=1)),
        composite_mean=(composite_mean),
        composite_std=(composite_std),
        composite_sem=(composite_sem),
        selection_score=(selection_score),
        fold_metrics=fold_metrics,
    )
