"""Cross-fit temperature calibration for PyTorch Linear streams."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from lung_fusion_agent.training.linear_classifier import (
    calculate_macro_auroc,
)

CACHE_PATH = Path("artifacts/results/linear_fusion_agent/linear_oof_predictions.npz")

OUTPUT_DIRECTORY = Path("artifacts/results/temperature_calibration")

EXPECTED_STREAMS = (
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
)

# Previously selected PyTorch Linear
# log-fusion weights.
GLOBAL_WEIGHTS = np.asarray(
    [
        0.05,
        0.15,
        0.10,
        0.70,
    ],
    dtype=np.float64,
)

TEMPERATURE_LOWER_BOUND = 0.25
TEMPERATURE_UPPER_BOUND = 4.0

FOLDS = 5
REPEATS = 3
SEED = 42

EPSILON = 1e-8


def stable_softmax(
    values: np.ndarray,
) -> np.ndarray:
    shifted = values - values.max(
        axis=1,
        keepdims=True,
    )

    exponentials = np.exp(shifted)

    return exponentials / exponentials.sum(
        axis=1,
        keepdims=True,
    )


def normalize_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        probabilities,
        EPSILON,
        1.0,
    )

    return clipped / clipped.sum(
        axis=-1,
        keepdims=True,
    )


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    probabilities = normalize_probabilities(probabilities)

    log_probabilities = np.log(probabilities)

    return stable_softmax(log_probabilities / temperature)


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    number_of_classes: int,
) -> float:
    probabilities = normalize_probabilities(probabilities)

    def objective(
        log_temperature: float,
    ) -> float:
        temperature = float(np.exp(log_temperature))

        calibrated = apply_temperature(
            probabilities,
            temperature,
        )

        return float(
            log_loss(
                labels,
                calibrated,
                labels=np.arange(number_of_classes),
            )
        )

    result = minimize_scalar(
        objective,
        bounds=(
            math.log(TEMPERATURE_LOWER_BOUND),
            math.log(TEMPERATURE_UPPER_BOUND),
        ),
        method="bounded",
        options={
            "xatol": 1e-6,
        },
    )

    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")

    temperature = float(np.exp(result.x))

    if not np.isfinite(temperature):
        raise RuntimeError("Temperature is non-finite")

    return temperature


def fit_stream_temperatures(
    probabilities: np.ndarray,
    labels: np.ndarray,
    number_of_classes: int,
) -> np.ndarray:
    if probabilities.ndim != 3:
        raise ValueError(
            "Expected probabilities with shape (patients, streams, classes)"
        )

    temperatures = np.empty(
        probabilities.shape[1],
        dtype=np.float64,
    )

    for stream_index in range(probabilities.shape[1]):
        temperatures[stream_index] = fit_temperature(
            probabilities[
                :,
                stream_index,
                :,
            ],
            labels,
            number_of_classes,
        )

    return temperatures


def apply_stream_temperatures(
    probabilities: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    calibrated = np.empty_like(
        probabilities,
        dtype=np.float64,
    )

    for stream_index, temperature in enumerate(temperatures):
        calibrated[
            :,
            stream_index,
            :,
        ] = apply_temperature(
            probabilities[
                :,
                stream_index,
                :,
            ],
            float(temperature),
        )

    return calibrated


def probability_fusion(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    fused = np.einsum(
        "m,nmc->nc",
        weights,
        probabilities,
    )

    return normalize_probabilities(fused)


def log_probability_fusion(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    probabilities = normalize_probabilities(probabilities)

    fused_log_values = np.einsum(
        "m,nmc->nc",
        weights,
        np.log(probabilities),
    )

    return stable_softmax(fused_log_values)


def evaluate_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    number_of_classes: int,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)

    macro_auroc = calculate_macro_auroc(
        labels,
        probabilities,
        number_of_classes,
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            labels,
            predictions,
        )
    )

    negative_log_likelihood = float(
        log_loss(
            labels,
            probabilities,
            labels=np.arange(number_of_classes),
        )
    )

    return {
        "macro_auroc": macro_auroc,
        "balanced_accuracy": (balanced_accuracy),
        "composite": (macro_auroc + balanced_accuracy) / 2,
        "negative_log_likelihood": (negative_log_likelihood),
    }


def standard_error(
    values: pd.Series,
) -> float:
    if len(values) < 2:
        return 0.0

    return float(values.std(ddof=1) / math.sqrt(len(values)))


def summarize_method(
    group: pd.DataFrame,
) -> dict[str, object]:
    composite_sem = standard_error(group["composite"])

    return {
        "method": str(group["method"].iloc[0]),
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
            "Expected probability cache with "
            "shape "
            "(repeats, patients, streams, classes)"
        )

    (
        number_of_repeats,
        number_of_patients,
        number_of_streams,
        number_of_classes,
    ) = probabilities.shape

    if number_of_repeats != REPEATS:
        raise ValueError(f"Expected {REPEATS} repeats, found {number_of_repeats}")

    if number_of_patients != 141:
        raise ValueError("Expected 141 training patients")

    if tuple(stream_names) != EXPECTED_STREAMS:
        raise ValueError(f"Unexpected stream order: {stream_names}")

    if number_of_streams != len(GLOBAL_WEIGHTS):
        raise ValueError("Fusion weights do not match the stream count")

    if len(labels) != number_of_patients:
        raise ValueError("Labels and probabilities differ")

    if not np.isfinite(probabilities).all():
        raise ValueError("OOF probabilities contain non-finite values")

    probabilities = normalize_probabilities(probabilities)

    splitter = RepeatedStratifiedKFold(
        n_splits=FOLDS,
        n_repeats=REPEATS,
        random_state=SEED,
    )

    # Only the indices are required here.
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

    if len(folds) != (FOLDS * REPEATS):
        raise RuntimeError("Unexpected number of calibration folds")

    fold_records: list[dict[str, object]] = []

    temperature_records: list[dict[str, object]] = []

    methods = (
        "uncalibrated_probability",
        "calibrated_probability",
        "uncalibrated_log_probability",
        "calibrated_log_probability",
    )

    for global_fold_index, (
        calibration_train_indices,
        calibration_validation_indices,
    ) in enumerate(folds):
        repeat_index = global_fold_index // FOLDS

        fold_index = global_fold_index % FOLDS

        repeat_probabilities = probabilities[repeat_index]

        train_probabilities = repeat_probabilities[calibration_train_indices]

        validation_probabilities = repeat_probabilities[calibration_validation_indices]

        train_labels = labels[calibration_train_indices]

        validation_labels = labels[calibration_validation_indices]

        temperatures = fit_stream_temperatures(
            train_probabilities,
            train_labels,
            number_of_classes,
        )

        calibrated_validation = apply_stream_temperatures(
            validation_probabilities,
            temperatures,
        )

        fused_outputs = {
            "uncalibrated_probability": (
                probability_fusion(
                    validation_probabilities,
                    GLOBAL_WEIGHTS,
                )
            ),
            "calibrated_probability": (
                probability_fusion(
                    calibrated_validation,
                    GLOBAL_WEIGHTS,
                )
            ),
            "uncalibrated_log_probability": (
                log_probability_fusion(
                    validation_probabilities,
                    GLOBAL_WEIGHTS,
                )
            ),
            "calibrated_log_probability": (
                log_probability_fusion(
                    calibrated_validation,
                    GLOBAL_WEIGHTS,
                )
            ),
        }

        for method in methods:
            metrics = evaluate_probabilities(
                fused_outputs[method],
                validation_labels,
                number_of_classes,
            )

            fold_records.append(
                {
                    "method": method,
                    "repeat_index": (repeat_index),
                    "fold_index": (fold_index),
                    "global_fold_index": (global_fold_index),
                    "calibration_train_records": (len(calibration_train_indices)),
                    "validation_records": (len(calibration_validation_indices)),
                    **metrics,
                }
            )

        for stream_index, stream_name in enumerate(stream_names):
            temperature_records.append(
                {
                    "repeat_index": (repeat_index),
                    "fold_index": (fold_index),
                    "global_fold_index": (global_fold_index),
                    "stream": stream_name,
                    "temperature": float(temperatures[stream_index]),
                }
            )

        print(f"Completed calibration split {global_fold_index + 1:02d}/{len(folds)}")

    fold_frame = pd.DataFrame(fold_records)

    fold_frame.to_csv(
        OUTPUT_DIRECTORY / "calibration_folds.csv",
        index=False,
    )

    temperature_frame = pd.DataFrame(temperature_records)

    temperature_frame.to_csv(
        OUTPUT_DIRECTORY / "fold_temperatures.csv",
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
        OUTPUT_DIRECTORY / "calibration_summary.csv",
        index=False,
    )

    # Fit deployable temperatures on one
    # averaged OOF prediction per patient.
    mean_patient_probabilities = probabilities.mean(axis=0)

    final_temperatures = fit_stream_temperatures(
        mean_patient_probabilities,
        labels,
        number_of_classes,
    )

    final_temperature_mapping = {
        stream_name: float(final_temperatures[stream_index])
        for stream_index, stream_name in enumerate(stream_names)
    }

    best_method = str(summary.iloc[0]["method"])

    result = {
        "classifier_family": ("PyTorch Linear"),
        "source_predictions": ("train-only out-of-fold"),
        "stream_order": stream_names,
        "class_names": class_names,
        "global_weights": {
            stream_name: float(GLOBAL_WEIGHTS[stream_index])
            for stream_index, stream_name in enumerate(stream_names)
        },
        "final_temperatures": (final_temperature_mapping),
        "temperature_bounds": [
            TEMPERATURE_LOWER_BOUND,
            TEMPERATURE_UPPER_BOUND,
        ],
        "best_method": best_method,
        "best_selection_score": float(summary.iloc[0]["selection_score"]),
        "folds": FOLDS,
        "repeats": REPEATS,
        "total_splits": len(folds),
        "validation_set_evaluated": False,
        "test_set_evaluated": False,
    }

    (OUTPUT_DIRECTORY / "selected_calibration.json").write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=== Temperature-calibration train-OOF results ===")

    print(
        summary[
            [
                "method",
                "macro_auroc_mean",
                "macro_auroc_std",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
                "negative_log_likelihood_mean",
                "selection_score",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== Final train-OOF temperatures ===")

    for stream_name, temperature in final_temperature_mapping.items():
        print(f"{stream_name}: {temperature:.6f}")

    print()
    print(
        "Selected calibration method:",
        best_method,
    )

    print()
    print("Fixed validation was not evaluated.")
    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
