"""Evaluate regularized class-specific log fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_temperature_calibration import (
    CACHE_PATH,
    EXPECTED_STREAMS,
    FOLDS,
    GLOBAL_WEIGHTS,
    REPEATS,
    SEED,
    evaluate_probabilities,
    normalize_probabilities,
    stable_softmax,
    standard_error,
)
from scipy.optimize import minimize
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)

OUTPUT_DIRECTORY = Path("artifacts/results/class_specific_fusion")

REGULARIZATION_STRENGTHS = (
    0.1,
    1.0,
    10.0,
    100.0,
)

MINIMUM_MATERIAL_IMPROVEMENT = 0.001

PARAMETER_LOWER_BOUND = -6.0
PARAMETER_UPPER_BOUND = 6.0

EPSILON = 1e-8


def parameters_to_weights(
    parameters: np.ndarray,
    number_of_streams: int,
    number_of_classes: int,
) -> np.ndarray:
    logits = parameters.reshape(
        number_of_streams,
        number_of_classes,
    )

    shifted = logits - logits.max(
        axis=0,
        keepdims=True,
    )

    exponentials = np.exp(shifted)

    weights = exponentials / exponentials.sum(
        axis=0,
        keepdims=True,
    )

    return weights


def class_specific_log_fusion(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if probabilities.ndim != 3:
        raise ValueError(
            "Expected probabilities with shape (patients, streams, classes)"
        )

    if weights.shape != (
        probabilities.shape[1],
        probabilities.shape[2],
    ):
        raise ValueError("Class-specific weight shape does not match probabilities")

    probabilities = normalize_probabilities(probabilities)

    fused_log_values = np.einsum(
        "nmc,mc->nc",
        np.log(
            np.clip(
                probabilities,
                EPSILON,
                1.0,
            )
        ),
        weights,
    )

    return stable_softmax(fused_log_values)


def global_log_fusion(
    probabilities: np.ndarray,
) -> np.ndarray:
    weights = np.repeat(
        GLOBAL_WEIGHTS[:, None],
        probabilities.shape[2],
        axis=1,
    )

    return class_specific_log_fusion(
        probabilities,
        weights,
    )


def fit_class_specific_weights(
    probabilities: np.ndarray,
    labels: np.ndarray,
    regularization_strength: float,
) -> tuple[
    np.ndarray,
    bool,
    float,
]:
    if regularization_strength < 0:
        raise ValueError("Regularization strength cannot be negative")

    probabilities = normalize_probabilities(probabilities)

    (
        number_of_patients,
        number_of_streams,
        number_of_classes,
    ) = probabilities.shape

    if len(labels) != number_of_patients:
        raise ValueError("Labels and probabilities differ")

    initial_logits = np.log(
        np.clip(
            GLOBAL_WEIGHTS,
            EPSILON,
            None,
        )
    )

    initial_parameters = np.repeat(
        initial_logits[:, None],
        number_of_classes,
        axis=1,
    ).reshape(-1)

    global_weight_matrix = np.repeat(
        GLOBAL_WEIGHTS[:, None],
        number_of_classes,
        axis=1,
    )

    patient_indices = np.arange(number_of_patients)

    def objective(
        parameters: np.ndarray,
    ) -> float:
        weights = parameters_to_weights(
            parameters,
            number_of_streams,
            number_of_classes,
        )

        fused = class_specific_log_fusion(
            probabilities,
            weights,
        )

        correct_probabilities = np.clip(
            fused[
                patient_indices,
                labels,
            ],
            EPSILON,
            1.0,
        )

        negative_log_likelihood = float(-np.log(correct_probabilities).mean())

        shrinkage_penalty = float(np.mean((weights - global_weight_matrix) ** 2))

        return negative_log_likelihood + regularization_strength * shrinkage_penalty

    result = minimize(
        objective,
        initial_parameters,
        method="L-BFGS-B",
        bounds=[
            (
                PARAMETER_LOWER_BOUND,
                PARAMETER_UPPER_BOUND,
            )
            for _ in initial_parameters
        ],
        options={
            "maxiter": 2000,
            "ftol": 1e-10,
            "gtol": 1e-7,
        },
    )

    weights = parameters_to_weights(
        result.x,
        number_of_streams,
        number_of_classes,
    )

    if not np.isfinite(weights).all():
        raise RuntimeError("Optimized weights contain non-finite values")

    if not np.allclose(
        weights.sum(axis=0),
        1.0,
        atol=1e-6,
    ):
        raise RuntimeError("Class-specific weights do not sum to one")

    return (
        weights,
        bool(result.success),
        float(result.fun),
    )


def summarize_method(
    group: pd.DataFrame,
) -> dict[str, object]:
    composite_sem = standard_error(group["composite"])

    regularization = group["regularization_strength"].iloc[0]

    if pd.isna(regularization):
        regularization_value = None
    else:
        regularization_value = float(regularization)

    return {
        "method": str(group["method"].iloc[0]),
        "regularization_strength": (regularization_value),
        "macro_auroc_mean": float(group["macro_auroc"].mean()),
        "macro_auroc_std": float(group["macro_auroc"].std(ddof=1)),
        "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(group["balanced_accuracy"].std(ddof=1)),
        "composite_mean": float(group["composite"].mean()),
        "composite_std": float(group["composite"].std(ddof=1)),
        "composite_sem": (composite_sem),
        "selection_score": float(group["composite"].mean() - 0.5 * composite_sem),
        "negative_log_likelihood_mean": (
            float(group["negative_log_likelihood"].mean())
        ),
        "optimization_success_rate": (float(group["optimization_success"].mean())),
    }


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing OOF cache: {CACHE_PATH}")

    with np.load(
        CACHE_PATH,
        allow_pickle=True,
    ) as cache:
        required_keys = {
            "probabilities",
            "labels",
            "stream_names",
            "class_names",
        }

        missing_keys = required_keys - set(cache.files)

        if missing_keys:
            raise ValueError(f"OOF cache missing keys: {sorted(missing_keys)}")

        probabilities = cache["probabilities"].astype(np.float64)

        labels = cache["labels"].astype(np.int64)

        stream_names = cache["stream_names"].astype(str).tolist()

        class_names = cache["class_names"].astype(str).tolist()

    if probabilities.ndim != 4:
        raise ValueError(
            "Expected probabilities with shape (repeats, patients, streams, classes)"
        )

    (
        number_of_repeats,
        number_of_patients,
        number_of_streams,
        number_of_classes,
    ) = probabilities.shape

    if number_of_repeats != REPEATS:
        raise ValueError(f"Expected {REPEATS} repeats")

    if number_of_patients != 141:
        raise ValueError("Expected 141 training patients")

    if tuple(stream_names) != (EXPECTED_STREAMS):
        raise ValueError(f"Unexpected stream order: {stream_names}")

    if number_of_streams != len(GLOBAL_WEIGHTS):
        raise ValueError("Global weights do not match the number of streams")

    probabilities = normalize_probabilities(probabilities)

    splitter = RepeatedStratifiedKFold(
        n_splits=FOLDS,
        n_repeats=REPEATS,
        random_state=SEED,
    )

    dummy_features = np.zeros(
        (number_of_patients, 1),
        dtype=np.float32,
    )

    folds = list(
        splitter.split(
            dummy_features,
            labels,
        )
    )

    if len(folds) != FOLDS * REPEATS:
        raise RuntimeError("Unexpected number of folds")

    fold_records: list[dict[str, object]] = []

    weight_records: list[dict[str, object]] = []

    for global_fold_index, (
        fusion_train_indices,
        fusion_validation_indices,
    ) in enumerate(folds):
        repeat_index = global_fold_index // FOLDS

        fold_index = global_fold_index % FOLDS

        repeat_probabilities = probabilities[repeat_index]

        train_probabilities = repeat_probabilities[fusion_train_indices]

        validation_probabilities = repeat_probabilities[fusion_validation_indices]

        train_labels = labels[fusion_train_indices]

        validation_labels = labels[fusion_validation_indices]

        baseline_probabilities = global_log_fusion(validation_probabilities)

        baseline_metrics = evaluate_probabilities(
            baseline_probabilities,
            validation_labels,
            number_of_classes,
        )

        fold_records.append(
            {
                "method": ("global_log_fusion"),
                "regularization_strength": (np.nan),
                "repeat_index": (repeat_index),
                "fold_index": (fold_index),
                "global_fold_index": (global_fold_index),
                "fusion_train_records": (len(fusion_train_indices)),
                "validation_records": (len(fusion_validation_indices)),
                "optimization_success": (True),
                "optimization_objective": (np.nan),
                **baseline_metrics,
            }
        )

        for regularization_strength in REGULARIZATION_STRENGTHS:
            (
                weights,
                optimization_success,
                optimization_objective,
            ) = fit_class_specific_weights(
                train_probabilities,
                train_labels,
                regularization_strength,
            )

            fused_probabilities = class_specific_log_fusion(
                validation_probabilities,
                weights,
            )

            metrics = evaluate_probabilities(
                fused_probabilities,
                validation_labels,
                number_of_classes,
            )

            method_name = f"class_specific_lambda_{regularization_strength:g}"

            fold_records.append(
                {
                    "method": method_name,
                    "regularization_strength": (regularization_strength),
                    "repeat_index": (repeat_index),
                    "fold_index": (fold_index),
                    "global_fold_index": (global_fold_index),
                    "fusion_train_records": (len(fusion_train_indices)),
                    "validation_records": (len(fusion_validation_indices)),
                    "optimization_success": (optimization_success),
                    "optimization_objective": (optimization_objective),
                    **metrics,
                }
            )

            for stream_index, stream_name in enumerate(stream_names):
                for class_index, class_name in enumerate(class_names):
                    weight_records.append(
                        {
                            "method": (method_name),
                            "regularization_strength": (regularization_strength),
                            "repeat_index": (repeat_index),
                            "fold_index": (fold_index),
                            "global_fold_index": (global_fold_index),
                            "stream": (stream_name),
                            "class_name": (class_name),
                            "weight": float(
                                weights[
                                    stream_index,
                                    class_index,
                                ]
                            ),
                        }
                    )

        print(f"Completed fusion split {global_fold_index + 1:02d}/{len(folds)}")

    fold_frame = pd.DataFrame(fold_records)

    fold_frame.to_csv(
        OUTPUT_DIRECTORY / "class_specific_folds.csv",
        index=False,
    )

    weight_frame = pd.DataFrame(weight_records)

    weight_frame.to_csv(
        OUTPUT_DIRECTORY / "fold_class_weights.csv",
        index=False,
    )

    summary_records = [
        summarize_method(group)
        for _, group in fold_frame.groupby(
            "method",
            sort=False,
        )
    ]

    summary = (
        pd.DataFrame(summary_records)
        .sort_values(
            "selection_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        OUTPUT_DIRECTORY / "class_specific_summary.csv",
        index=False,
    )

    baseline_row = summary.loc[summary["method"].eq("global_log_fusion")].iloc[0]

    class_specific_rows = (
        summary.loc[summary["method"].str.startswith("class_specific_")]
        .sort_values(
            "selection_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_class_specific = class_specific_rows.iloc[0]

    improvement = float(
        best_class_specific["selection_score"] - baseline_row["selection_score"]
    )

    materially_improved = bool(improvement >= MINIMUM_MATERIAL_IMPROVEMENT)

    best_regularization = float(best_class_specific["regularization_strength"])

    # Fit deployable weights using one
    # averaged OOF prediction per patient.
    mean_patient_probabilities = probabilities.mean(axis=0)

    (
        final_weights,
        final_optimization_success,
        final_optimization_objective,
    ) = fit_class_specific_weights(
        mean_patient_probabilities,
        labels,
        best_regularization,
    )

    final_weight_records: list[dict[str, object]] = []

    for stream_index, stream_name in enumerate(stream_names):
        for class_index, class_name in enumerate(class_names):
            final_weight_records.append(
                {
                    "stream": stream_name,
                    "class_name": class_name,
                    "weight": float(
                        final_weights[
                            stream_index,
                            class_index,
                        ]
                    ),
                }
            )

    final_weight_frame = pd.DataFrame(final_weight_records)

    final_weight_frame.to_csv(
        OUTPUT_DIRECTORY / "final_class_weights.csv",
        index=False,
    )

    final_weight_mapping = {
        class_name: {
            stream_name: float(
                final_weights[
                    stream_index,
                    class_index,
                ]
            )
            for stream_index, stream_name in enumerate(stream_names)
        }
        for class_index, class_name in enumerate(class_names)
    }

    selected_method = (
        str(best_class_specific["method"])
        if materially_improved
        else "global_log_fusion"
    )

    result = {
        "classifier_family": ("PyTorch Linear"),
        "source_predictions": ("train-only out-of-fold"),
        "stream_order": stream_names,
        "class_names": class_names,
        "global_weights": {
            stream_name: float(GLOBAL_WEIGHTS[stream_index])
            for stream_index, stream_name in enumerate(stream_names)
        },
        "regularization_trial_budget": len(REGULARIZATION_STRENGTHS),
        "regularization_strengths": list(REGULARIZATION_STRENGTHS),
        "best_class_specific_method": str(best_class_specific["method"]),
        "best_regularization_strength": (best_regularization),
        "global_selection_score": float(baseline_row["selection_score"]),
        "best_class_specific_selection_score": (
            float(best_class_specific["selection_score"])
        ),
        "selection_score_improvement": (improvement),
        "minimum_material_improvement": (MINIMUM_MATERIAL_IMPROVEMENT),
        "materially_improved": (materially_improved),
        "selected_method": (selected_method),
        "final_optimization_success": (final_optimization_success),
        "final_optimization_objective": (final_optimization_objective),
        "final_class_specific_weights": (final_weight_mapping),
        "folds": FOLDS,
        "repeats": REPEATS,
        "total_splits": len(folds),
        "validation_set_evaluated": False,
        "test_set_evaluated": False,
    }

    (OUTPUT_DIRECTORY / "selected_class_specific.json").write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=== Class-specific fusion train-OOF results ===")

    print(
        summary[
            [
                "method",
                "regularization_strength",
                "macro_auroc_mean",
                "balanced_accuracy_mean",
                "composite_mean",
                "composite_std",
                "selection_score",
                "optimization_success_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Global selection score:",
        round(
            float(baseline_row["selection_score"]),
            6,
        ),
    )

    print(
        "Best class-specific score:",
        round(
            float(best_class_specific["selection_score"]),
            6,
        ),
    )

    print(
        "Improvement:",
        round(
            improvement,
            6,
        ),
    )

    print(
        "Materially improved:",
        materially_improved,
    )

    print(
        "Selected method:",
        selected_method,
    )

    print()
    print("Fixed validation was not evaluated.")
    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
