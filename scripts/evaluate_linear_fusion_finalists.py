"""Evaluate locked Linear late-fusion finalists on fixed validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)

LINEAR_VALIDATION_ROOT = Path("artifacts/results/linear_validation")

FUSION_ROOT = Path("artifacts/results/linear_fusion_agent")

OUTPUT_ROOT = Path("artifacts/results/linear_fusion_finalists")

STREAMS = (
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
)


def load_weights(
    method: str,
) -> np.ndarray:
    path = FUSION_ROOT / f"late_fusion_{method}_best.json"

    result = json.loads(path.read_text())

    if result["validation_set_evaluated"]:
        raise ValueError(f"{method} weights used fixed validation")

    if result["test_set_evaluated"]:
        raise ValueError(f"{method} weights used test")

    if result["stream_order"] != list(STREAMS):
        raise ValueError("Unexpected stream order")

    weights = np.asarray(
        result["best_trial"]["weights"],
        dtype=np.float64,
    )

    if weights.shape != (len(STREAMS),):
        raise ValueError("Unexpected weight shape")

    if np.any(weights < 0):
        raise ValueError("Fusion weights cannot be negative")

    np.testing.assert_allclose(
        weights.sum(),
        1.0,
        atol=1e-8,
    )

    return weights


def load_validation_probabilities() -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    reference = None
    probability_arrays = []
    class_names = None

    for stream in STREAMS:
        path = LINEAR_VALIDATION_ROOT / stream / "validation_predictions.csv"

        frame = pd.read_csv(path)

        probability_columns = [
            column for column in frame.columns if column.startswith("probability_")
        ]

        current_class_names = [
            column.removeprefix("probability_") for column in probability_columns
        ]

        if class_names is None:
            class_names = current_class_names

        elif current_class_names != class_names:
            raise ValueError(f"Class order differs for {stream}")

        current_identity = frame[
            [
                "patient_id",
                "true_label",
            ]
        ].copy()

        if reference is None:
            reference = current_identity

        elif not current_identity.equals(reference):
            raise ValueError(f"Validation patient order differs for {stream}")

        probabilities = frame[probability_columns].to_numpy(dtype=np.float64)

        if not np.isfinite(probabilities).all():
            raise ValueError(f"{stream} contains non-finite probabilities")

        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-5,
        )

        probability_arrays.append(probabilities)

    if reference is None:
        raise RuntimeError("No validation predictions")

    if class_names is None:
        raise RuntimeError("No class names found")

    stacked = np.stack(
        probability_arrays,
        axis=1,
    )

    class_to_index = {
        class_name: class_index for class_index, class_name in (enumerate(class_names))
    }

    labels = np.asarray(
        [class_to_index[label] for label in (reference["true_label"].astype(str))],
        dtype=np.int64,
    )

    patient_ids = reference["patient_id"].astype(str).to_numpy(dtype=str)

    return (
        class_names,
        patient_ids,
        labels,
        stacked,
    )


def probability_fusion(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    fused = np.sum(
        probabilities
        * weights[
            None,
            :,
            None,
        ],
        axis=1,
    )

    return fused / fused.sum(
        axis=1,
        keepdims=True,
    )


def log_probability_fusion(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    safe_probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    scores = np.sum(
        np.log(safe_probabilities)
        * weights[
            None,
            :,
            None,
        ],
        axis=1,
    )

    # Numerically stable softmax.
    scores = scores - scores.max(
        axis=1,
        keepdims=True,
    )

    exponentiated = np.exp(scores)

    return exponentiated / exponentiated.sum(
        axis=1,
        keepdims=True,
    )


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    number_of_classes: int,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)

    macro_auroc = roc_auc_score(
        labels,
        probabilities,
        labels=np.arange(number_of_classes),
        multi_class="ovr",
        average="macro",
    )

    balanced_accuracy = balanced_accuracy_score(
        labels,
        predictions,
    )

    loss = log_loss(
        labels,
        probabilities,
        labels=np.arange(number_of_classes),
    )

    return {
        "macro_auroc": float(macro_auroc),
        "balanced_accuracy": float(balanced_accuracy),
        "validation_loss": float(loss),
        "composite": float((macro_auroc + balanced_accuracy) / 2),
    }


def main() -> None:
    (
        class_names,
        patient_ids,
        labels,
        stream_probabilities,
    ) = load_validation_probabilities()

    probability_weights = load_weights("probability")

    logit_weights = load_weights("logit")

    probability_predictions = probability_fusion(
        stream_probabilities,
        probability_weights,
    )

    logit_predictions = log_probability_fusion(
        stream_probabilities,
        logit_weights,
    )

    # Include the strongest single stream
    # as a validation reference.
    prism2_index = STREAMS.index("prism2")

    prism2_predictions = stream_probabilities[
        :,
        prism2_index,
        :,
    ]

    candidates = {
        "prism2": (prism2_predictions),
        "probability_fusion": (probability_predictions),
        "log_probability_fusion": (logit_predictions),
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_records = []

    for name, probabilities in candidates.items():
        metrics = calculate_metrics(
            labels,
            probabilities,
            len(class_names),
        )

        result_records.append(
            {
                "strategy": name,
                **metrics,
            }
        )

        predicted_indices = probabilities.argmax(axis=1)

        prediction_table = pd.DataFrame(
            {
                "patient_id": (patient_ids),
                "true_label": [class_names[index] for index in labels],
                "predicted_label": [
                    class_names[index] for index in (predicted_indices)
                ],
            }
        )

        for class_index, class_name in enumerate(class_names):
            prediction_table[f"probability_{class_name}"] = probabilities[
                :,
                class_index,
            ]

        prediction_table.to_csv(
            OUTPUT_ROOT / f"{name}_predictions.csv",
            index=False,
        )

    results = (
        pd.DataFrame(result_records)
        .sort_values(
            "composite",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    results.to_csv(
        OUTPUT_ROOT / "validation_results.csv",
        index=False,
    )

    fusion_results = results.loc[
        results["strategy"].isin(
            (
                "probability_fusion",
                "log_probability_fusion",
            )
        )
    ]

    selected_fusion = str(fusion_results.iloc[0]["strategy"])

    selection = {
        "selected_fusion": (selected_fusion),
        "selection_metric": ("(macro AUROC + balanced accuracy) / 2"),
        "probability_weights": (probability_weights.tolist()),
        "log_probability_weights": (logit_weights.tolist()),
        "stream_order": list(STREAMS),
        "validation_records": len(labels),
        "test_set_evaluated": False,
    }

    (OUTPUT_ROOT / "selected_fusion.json").write_text(
        json.dumps(
            selection,
            indent=2,
        )
        + "\n"
    )

    print("=== Linear fusion fixed-validation results ===")

    print(results.to_string(index=False))

    print()
    print(
        "Selected fusion:",
        selected_fusion,
    )

    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
