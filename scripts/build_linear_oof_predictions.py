"""Build OOF probabilities from selected PyTorch Linear classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_linear_training_curves import (
    build_features,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

from lung_fusion_agent.training import (
    LinearTrainingConfig,
    choose_device,
    predict_probabilities,
    train_linear_classifier,
)

SPLIT_PATH = Path("artifacts/splits/patient_splits.csv")

CONFIG_ROOT = Path("artifacts/results/linear_cv")

OUTPUT_DIRECTORY = Path("artifacts/results/linear_fusion_agent")

STREAMS = (
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
)

NUMBER_OF_FOLDS = 5
NUMBER_OF_REPEATS = 3
SEED = 42


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

    composite = (macro_auroc + balanced_accuracy) / 2

    return {
        "macro_auroc": float(macro_auroc),
        "balanced_accuracy": float(balanced_accuracy),
        "composite": float(composite),
    }


def load_selected_config(
    stream: str,
) -> dict[str, object]:
    path = CONFIG_ROOT / stream / "selected_config.json"

    if not path.exists():
        raise FileNotFoundError(path)

    selected = json.loads(path.read_text())

    if selected["representation"] != stream:
        raise ValueError(f"Configuration mismatch for {stream}")

    if selected["validation_set_evaluated"]:
        raise ValueError(f"{stream} configuration was not selected using train-only CV")

    if selected["test_set_evaluated"]:
        raise ValueError(f"{stream} configuration used the test set")

    return selected


def main() -> None:
    split_table = pd.read_csv(SPLIT_PATH)

    split_table["patient_id"] = split_table["patient_id"].astype(str)

    split_table["split"] = split_table["split"].astype(str).str.lower()

    # Only the original 141 training
    # records enter OOF generation.
    train_table = split_table.loc[split_table["split"].eq("train")].reset_index(
        drop=True
    )

    if len(train_table) != 141:
        raise ValueError("Expected exactly 141 training records")

    if train_table["patient_id"].duplicated().any():
        raise ValueError("Training table contains duplicate patient IDs")

    label_encoder = LabelEncoder()

    labels = label_encoder.fit_transform(train_table["label"].astype(str))

    class_names = label_encoder.classes_.astype(str)

    number_of_classes = len(class_names)

    splitter = RepeatedStratifiedKFold(
        n_splits=NUMBER_OF_FOLDS,
        n_repeats=NUMBER_OF_REPEATS,
        random_state=SEED,
    )

    # Generate fold indices once so all
    # streams use identical held-out patients.
    placeholder_features = np.zeros(
        (
            len(train_table),
            1,
        ),
        dtype=np.float32,
    )

    folds = list(
        splitter.split(
            placeholder_features,
            labels,
        )
    )

    number_of_splits = len(folds)

    if number_of_splits != 15:
        raise RuntimeError(f"Expected 15 splits, found {number_of_splits}")

    number_of_patients = len(train_table)

    number_of_streams = len(STREAMS)

    # Dense form:
    # repeat x patient x stream x class
    dense_probabilities = np.full(
        (
            NUMBER_OF_REPEATS,
            number_of_patients,
            number_of_streams,
            number_of_classes,
        ),
        np.nan,
        dtype=np.float32,
    )

    # Sparse split form preserves each of
    # the 15 individual held-out folds.
    split_probabilities = np.full(
        (
            number_of_splits,
            number_of_patients,
            number_of_streams,
            number_of_classes,
        ),
        np.nan,
        dtype=np.float32,
    )

    validation_masks = np.zeros(
        (
            number_of_splits,
            number_of_patients,
        ),
        dtype=bool,
    )

    fold_assignments = np.full(
        (
            NUMBER_OF_REPEATS,
            number_of_patients,
        ),
        -1,
        dtype=np.int16,
    )

    prediction_counts = np.zeros(
        (
            NUMBER_OF_REPEATS,
            number_of_patients,
            number_of_streams,
        ),
        dtype=np.int16,
    )

    fold_records: list[dict[str, object]] = []

    selected_configs = {stream: load_selected_config(stream) for stream in STREAMS}

    device = choose_device()

    print(f"Device: {device}")
    print(f"Training patients: {number_of_patients}")
    print(f"Streams: {list(STREAMS)}")
    print(f"CV splits: {number_of_splits}")
    print("Fixed validation and test were not loaded.")

    for stream_index, stream in enumerate(STREAMS):
        print()
        print("=" * 70)
        print(f"Stream: {stream}")

        selected = selected_configs[stream]

        print(
            json.dumps(
                {
                    "learning_rate": (selected["learning_rate"]),
                    "weight_decay": (selected["weight_decay"]),
                    "class_weight": (selected["class_weight"]),
                    "fixed_epochs": (selected["fixed_epochs"]),
                },
                indent=2,
            )
        )

        features = build_features(
            stream,
            train_table,
        )

        for global_fold_index, (
            fold_train_indices,
            fold_validation_indices,
        ) in enumerate(folds):
            repeat_index = global_fold_index // NUMBER_OF_FOLDS

            fold_index = global_fold_index % NUMBER_OF_FOLDS

            scaler = StandardScaler()

            fold_train_features = scaler.fit_transform(
                features[fold_train_indices]
            ).astype(np.float32)

            fold_validation_features = scaler.transform(
                features[fold_validation_indices]
            ).astype(np.float32)

            fold_train_labels = labels[fold_train_indices]

            fold_validation_labels = labels[fold_validation_indices]

            config = LinearTrainingConfig(
                epochs=int(selected["fixed_epochs"]),
                learning_rate=float(selected["learning_rate"]),
                weight_decay=float(selected["weight_decay"]),
                class_weight=str(selected["class_weight"]),
                seed=(int(selected["seed"]) + global_fold_index),
                early_stopping_patience=None,
                minimum_improvement=1e-3,
                restore_best_state=False,
            )

            result = train_linear_classifier(
                train_features=(fold_train_features),
                train_labels=(fold_train_labels),
                validation_features=(fold_validation_features),
                validation_labels=(fold_validation_labels),
                number_of_classes=(number_of_classes),
                config=config,
                device=device,
            )

            validation_tensor = torch.from_numpy(fold_validation_features).to(device)

            probabilities = predict_probabilities(
                result.model,
                validation_tensor,
            )

            if probabilities.shape != (
                len(fold_validation_indices),
                number_of_classes,
            ):
                raise ValueError(f"Unexpected probability shape: {probabilities.shape}")

            if not np.isfinite(probabilities).all():
                raise ValueError("Non-finite OOF probabilities")

            np.testing.assert_allclose(
                probabilities.sum(axis=1),
                1.0,
                atol=1e-5,
            )

            dense_probabilities[
                repeat_index,
                fold_validation_indices,
                stream_index,
                :,
            ] = probabilities

            split_probabilities[
                global_fold_index,
                fold_validation_indices,
                stream_index,
                :,
            ] = probabilities

            validation_masks[
                global_fold_index,
                fold_validation_indices,
            ] = True

            fold_assignments[
                repeat_index,
                fold_validation_indices,
            ] = fold_index

            prediction_counts[
                repeat_index,
                fold_validation_indices,
                stream_index,
            ] += 1

            metrics = calculate_metrics(
                fold_validation_labels,
                probabilities,
                number_of_classes,
            )

            fold_records.append(
                {
                    "stream": stream,
                    "repeat_index": (repeat_index),
                    "fold_index": (fold_index),
                    "global_fold_index": (global_fold_index),
                    "train_records": len(fold_train_indices),
                    "validation_records": (len(fold_validation_indices)),
                    "learning_rate": (config.learning_rate),
                    "weight_decay": (config.weight_decay),
                    "class_weight": (config.class_weight),
                    "fixed_epochs": (config.epochs),
                    **metrics,
                }
            )

            print(
                f"  repeat="
                f"{repeat_index + 1} "
                f"fold={fold_index + 1} "
                f"AUROC="
                f"{metrics['macro_auroc']:.4f} "
                f"BA="
                f"{metrics['balanced_accuracy']:.4f}"
            )

    if not np.all(prediction_counts == 1):
        unique_counts = np.unique(prediction_counts)

        raise RuntimeError(
            "Every patient must have "
            "one OOF prediction per "
            "repeat and stream. "
            f"Counts found: "
            f"{unique_counts.tolist()}"
        )

    if not np.isfinite(dense_probabilities).all():
        raise RuntimeError("Dense OOF cache contains missing values")

    for global_fold_index in range(number_of_splits):
        mask = validation_masks[global_fold_index]

        held_out_probabilities = split_probabilities[
            global_fold_index,
            mask,
            :,
            :,
        ]

        if not np.isfinite(held_out_probabilities).all():
            raise RuntimeError("Held-out split contains missing probabilities")

        outside_probabilities = split_probabilities[
            global_fold_index,
            ~mask,
            :,
            :,
        ]

        if not np.isnan(outside_probabilities).all():
            raise RuntimeError("Non-held-out patients must remain NaN")

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dense_path = OUTPUT_DIRECTORY / "linear_oof_predictions.npz"

    split_path = OUTPUT_DIRECTORY / "linear_oof_split_predictions.npz"

    np.savez_compressed(
        dense_path,
        probabilities=(dense_probabilities),
        patient_ids=np.asarray(
            train_table["patient_id"].astype(str).tolist(),
            dtype=str,
        ),
        labels=labels.astype(np.int64),
        class_names=class_names,
        stream_names=np.asarray(
            STREAMS,
            dtype=str,
        ),
        fold_assignments=(fold_assignments),
        seed=np.asarray(
            SEED,
            dtype=np.int64,
        ),
    )

    np.savez_compressed(
        split_path,
        probabilities=(split_probabilities),
        validation_masks=(validation_masks),
        patient_ids=np.asarray(
            train_table["patient_id"].astype(str).tolist(),
            dtype=str,
        ),
        labels=labels.astype(np.int64),
        class_names=class_names,
        stream_names=np.asarray(
            STREAMS,
            dtype=str,
        ),
        seed=np.asarray(
            SEED,
            dtype=np.int64,
        ),
    )

    fold_results = pd.DataFrame(fold_records)

    fold_results.to_csv(
        OUTPUT_DIRECTORY / "oof_fold_metrics.csv",
        index=False,
    )

    repeat_records = []

    for stream_index, stream in enumerate(STREAMS):
        for repeat_index in range(NUMBER_OF_REPEATS):
            probabilities = dense_probabilities[
                repeat_index,
                :,
                stream_index,
                :,
            ]

            metrics = calculate_metrics(
                labels,
                probabilities,
                number_of_classes,
            )

            repeat_records.append(
                {
                    "stream": stream,
                    "repeat_index": (repeat_index),
                    **metrics,
                }
            )

    repeat_results = pd.DataFrame(repeat_records)

    repeat_results.to_csv(
        OUTPUT_DIRECTORY / "oof_repeat_metrics.csv",
        index=False,
    )

    stream_summary = (
        repeat_results.groupby(
            "stream",
            sort=False,
        )
        .agg(
            macro_auroc_mean=(
                "macro_auroc",
                "mean",
            ),
            macro_auroc_std=(
                "macro_auroc",
                "std",
            ),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),
            composite_mean=(
                "composite",
                "mean",
            ),
            composite_std=(
                "composite",
                "std",
            ),
        )
        .reset_index()
    )

    stream_summary.to_csv(
        OUTPUT_DIRECTORY / "oof_stream_summary.csv",
        index=False,
    )

    provenance = {
        "seed": SEED,
        "folds": NUMBER_OF_FOLDS,
        "repeats": NUMBER_OF_REPEATS,
        "patients": number_of_patients,
        "classes": class_names.tolist(),
        "streams": list(STREAMS),
        "dense_probability_shape": list(dense_probabilities.shape),
        "split_probability_shape": list(split_probabilities.shape),
        "selected_configs": (selected_configs),
        "fixed_validation_evaluated": (False),
        "test_set_evaluated": False,
    }

    (OUTPUT_DIRECTORY / "oof_provenance.json").write_text(
        json.dumps(
            provenance,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=== Linear OOF summary ===")

    print(stream_summary.to_string(index=False))

    print()
    print(
        "Dense probability shape:",
        dense_probabilities.shape,
    )

    print(
        "Split probability shape:",
        split_probabilities.shape,
    )

    print(
        "Predictions per patient, repeat, and stream:",
        np.unique(prediction_counts).tolist(),
    )

    print(f"Dense cache: {dense_path}")
    print(f"Split cache: {split_path}")

    print("Fixed validation was not evaluated.")

    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
