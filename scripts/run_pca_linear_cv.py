"""Evaluate fold-local PCA early fusion with PyTorch Linear."""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from run_linear_cv import (
    CLASS_WEIGHTS,
    LEARNING_RATES,
    MODEL_FILES,
    SPLIT_PATH,
    WEIGHT_DECAYS,
    align_embedding,
    metadata_features,
    sem,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from lung_fusion_agent.training.linear_classifier import (
    LinearTrainingConfig,
    choose_device,
    train_linear_classifier,
)

MODEL_NAMES = tuple(MODEL_FILES)
PCA_COMPONENTS = 32

OUTPUT_DIRECTORY = Path("artifacts/results/pca_linear_cv/pca_early_concat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--minimum-improvement",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_DIRECTORY,
    )

    return parser.parse_args()


def load_training_data() -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    list[str],
]:
    split_table = pd.read_csv(SPLIT_PATH)

    required_columns = {
        "patient_id",
        "age",
        "sex",
        "label",
        "split",
    }

    missing_columns = required_columns - set(split_table.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    split_table["patient_id"] = split_table["patient_id"].astype(str)

    split_table["split"] = split_table["split"].astype(str).str.lower()

    train_table = split_table.loc[split_table["split"].eq("train")].reset_index(
        drop=True
    )

    if len(train_table) != 141:
        raise ValueError("Expected exactly 141 training records")

    patient_ids = train_table["patient_id"].astype(str).to_numpy()

    streams = {
        model_name: align_embedding(
            patient_ids,
            model_name,
        ).astype(np.float32)
        for model_name in MODEL_NAMES
    }

    metadata = metadata_features(train_table).astype(np.float32)

    label_encoder = LabelEncoder()

    labels = label_encoder.fit_transform(train_table["label"].astype(str)).astype(
        np.int64
    )

    class_names = label_encoder.classes_.astype(str).tolist()

    return (
        train_table,
        streams,
        metadata,
        labels,
        class_names,
    )


def prepare_fold_features(
    streams: dict[str, np.ndarray],
    metadata: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
]:
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []

    variance_records: list[dict[str, object]] = []

    for model_name in MODEL_NAMES:
        pca = PCA(
            n_components=PCA_COMPONENTS,
            svd_solver="full",
        )

        train_projection = pca.fit_transform(streams[model_name][train_indices]).astype(
            np.float32
        )

        validation_projection = pca.transform(
            streams[model_name][validation_indices]
        ).astype(np.float32)

        if not np.isfinite(train_projection).all():
            raise ValueError(
                f"{model_name} PCA train projection contains non-finite values"
            )

        if not np.isfinite(validation_projection).all():
            raise ValueError(
                f"{model_name} PCA validation projection contains non-finite values"
            )

        train_parts.append(train_projection)

        validation_parts.append(validation_projection)

        variance_records.append(
            {
                "model_name": model_name,
                "pca_components": (PCA_COMPONENTS),
                "explained_variance_ratio_sum": (
                    float(pca.explained_variance_ratio_.sum())
                ),
            }
        )

    train_parts.append(metadata[train_indices])

    validation_parts.append(metadata[validation_indices])

    train_fused = np.concatenate(
        train_parts,
        axis=1,
    )

    validation_fused = np.concatenate(
        validation_parts,
        axis=1,
    )

    expected_dimension = PCA_COMPONENTS * len(MODEL_NAMES) + metadata.shape[1]

    if train_fused.shape[1] != expected_dimension:
        raise RuntimeError(f"Unexpected PCA-fusion dimension: {train_fused.shape[1]}")

    # This scaler is also fitted only on the
    # fold-training patients.
    fused_scaler = StandardScaler()

    train_scaled = fused_scaler.fit_transform(train_fused).astype(np.float32)

    validation_scaled = fused_scaler.transform(validation_fused).astype(np.float32)

    return (
        train_scaled,
        validation_scaled,
        variance_records,
    )


def main() -> None:
    args = parse_args()

    (
        train_table,
        streams,
        metadata,
        labels,
        class_names,
    ) = load_training_data()

    splitter = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.seed,
    )

    folds = list(
        splitter.split(
            metadata,
            labels,
        )
    )

    if len(folds) != (args.folds * args.repeats):
        raise RuntimeError("Unexpected number of CV splits")

    search_space = list(
        product(
            LEARNING_RATES,
            WEIGHT_DECAYS,
            CLASS_WEIGHTS,
        )
    )

    if len(search_space) != 12:
        raise RuntimeError("Expected exactly 12 classifier configurations")

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = choose_device()

    fused_dimension = PCA_COMPONENTS * len(MODEL_NAMES) + metadata.shape[1]

    print("Representation: pca_early_concat")
    print("Classifier family: PyTorch Linear")
    print(f"Device: {device}")
    print(f"Training patients: {len(train_table)}")
    print(f"Foundation streams: {list(MODEL_NAMES)}")
    print(f"PCA components per stream: {PCA_COMPONENTS}")
    print(f"Fused dimension: {fused_dimension}")
    print(f"Classes: {class_names}")
    print(f"CV splits: {len(folds)}")
    print(f"Classifier configurations: {len(search_space)}")

    # PCA and StandardScaler are independent
    # of the classifier hyperparameters.
    # Compute each fold once and reuse it for
    # all 12 classifier trials.
    prepared_folds: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []

    variance_records: list[dict[str, object]] = []

    print()
    print("Preparing fold-local PCA features...")

    for (
        global_fold_index,
        (
            fold_train_indices,
            fold_validation_indices,
        ),
    ) in enumerate(folds):
        (
            fold_train_features,
            fold_validation_features,
            fold_variance_records,
        ) = prepare_fold_features(
            streams=streams,
            metadata=metadata,
            train_indices=(fold_train_indices),
            validation_indices=(fold_validation_indices),
        )

        prepared_folds.append(
            (
                fold_train_features,
                fold_validation_features,
                labels[fold_train_indices],
                labels[fold_validation_indices],
            )
        )

        repeat_index = global_fold_index // args.folds

        fold_index = global_fold_index % args.folds

        for record in fold_variance_records:
            variance_records.append(
                {
                    "repeat_index": (repeat_index),
                    "fold_index": (fold_index),
                    "global_fold_index": (global_fold_index),
                    **record,
                }
            )

        print(f"  prepared split {global_fold_index + 1:02d}/{len(folds)}")

    variance_frame = pd.DataFrame(variance_records)

    variance_frame.to_csv(
        args.output_directory / "pca_variance.csv",
        index=False,
    )

    fold_records: list[dict[str, object]] = []

    run_start = time.perf_counter()

    for (
        trial_index,
        (
            learning_rate,
            weight_decay,
            class_weight,
        ),
    ) in enumerate(search_space):
        print()
        print("=" * 70)
        print(f"Trial {trial_index + 1}/{len(search_space)}")
        print(f"learning_rate={learning_rate}")
        print(f"weight_decay={weight_decay}")
        print(f"class_weight={class_weight}")

        for (
            global_fold_index,
            (
                fold_train_features,
                fold_validation_features,
                fold_train_labels,
                fold_validation_labels,
            ),
        ) in enumerate(prepared_folds):
            repeat_index = global_fold_index // args.folds

            fold_index = global_fold_index % args.folds

            config = LinearTrainingConfig(
                epochs=args.max_epochs,
                learning_rate=(learning_rate),
                weight_decay=weight_decay,
                class_weight=class_weight,
                seed=(args.seed + trial_index * 1000 + global_fold_index),
                early_stopping_patience=(args.patience),
                minimum_improvement=(args.minimum_improvement),
                restore_best_state=True,
            )

            fold_start = time.perf_counter()

            result = train_linear_classifier(
                train_features=(fold_train_features),
                train_labels=(fold_train_labels),
                validation_features=(fold_validation_features),
                validation_labels=(fold_validation_labels),
                number_of_classes=len(class_names),
                config=config,
                device=device,
            )

            elapsed_seconds = time.perf_counter() - fold_start

            history = pd.DataFrame(result.history)

            best_rows = history.loc[history["epoch"].eq(result.best_epoch)]

            if len(best_rows) != 1:
                raise RuntimeError("Expected exactly one best-epoch history row")

            best_row = best_rows.iloc[0]

            fold_record = {
                "representation": ("pca_early_concat"),
                "classifier_family": ("linear"),
                "trial_index": (trial_index),
                "repeat_index": (repeat_index),
                "fold_index": (fold_index),
                "global_fold_index": (global_fold_index),
                "pca_components": (PCA_COMPONENTS),
                "fused_dimension": (fused_dimension),
                "learning_rate": (learning_rate),
                "weight_decay": (weight_decay),
                "class_weight": (class_weight),
                "train_records": len(fold_train_labels),
                "validation_records": len(fold_validation_labels),
                "best_epoch": (result.best_epoch),
                "epochs_completed": len(history),
                "stopped_early": (result.stopped_early),
                "validation_loss": float(best_row["validation_loss"]),
                "macro_auroc": float(best_row["validation_macro_auroc"]),
                "balanced_accuracy": float(best_row["validation_balanced_accuracy"]),
                "composite": float(best_row["validation_composite"]),
                "elapsed_seconds": (elapsed_seconds),
            }

            fold_records.append(fold_record)

            print(
                f"  split "
                f"{global_fold_index + 1:02d}/"
                f"{len(folds)} | "
                f"epoch="
                f"{result.best_epoch:03d} | "
                f"AUROC="
                f"{fold_record['macro_auroc']:.4f} | "
                f"BA="
                f"{fold_record['balanced_accuracy']:.4f}"
            )

        pd.DataFrame(fold_records).to_csv(
            args.output_directory / "cv_folds.csv",
            index=False,
        )

    fold_results = pd.DataFrame(fold_records)

    trial_records: list[dict[str, object]] = []

    for (
        trial_index,
        group,
    ) in fold_results.groupby(
        "trial_index",
        sort=True,
    ):
        composite_sem = sem(group["composite"])

        selection_score = group["composite"].mean() - 0.5 * composite_sem

        trial_records.append(
            {
                "trial_index": int(trial_index),
                "pca_components": (PCA_COMPONENTS),
                "fused_dimension": (fused_dimension),
                "learning_rate": float(group["learning_rate"].iloc[0]),
                "weight_decay": float(group["weight_decay"].iloc[0]),
                "class_weight": str(group["class_weight"].iloc[0]),
                "macro_auroc_mean": (group["macro_auroc"].mean()),
                "macro_auroc_std": (group["macro_auroc"].std(ddof=1)),
                "balanced_accuracy_mean": (group["balanced_accuracy"].mean()),
                "balanced_accuracy_std": (group["balanced_accuracy"].std(ddof=1)),
                "composite_mean": (group["composite"].mean()),
                "composite_std": (group["composite"].std(ddof=1)),
                "composite_sem": (composite_sem),
                "selection_score": (selection_score),
                "median_best_epoch": round(group["best_epoch"].median()),
                "mean_epochs_completed": (group["epochs_completed"].mean()),
                "mean_elapsed_seconds": (group["elapsed_seconds"].mean()),
            }
        )

    trial_results = (
        pd.DataFrame(trial_records)
        .sort_values(
            "selection_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    trial_results.to_csv(
        args.output_directory / "cv_trials.csv",
        index=False,
    )

    best_trial = trial_results.iloc[0]

    variance_summary = (
        variance_frame.groupby("model_name")["explained_variance_ratio_sum"]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    variance_summary.to_csv(
        args.output_directory / "pca_variance_summary.csv",
        index=False,
    )

    selected_config = {
        "representation": ("pca_early_concat"),
        "classifier_family": ("linear"),
        "pca_scope": ("fit separately inside each CV training fold"),
        "pca_components_per_stream": (PCA_COMPONENTS),
        "foundation_streams": list(MODEL_NAMES),
        "metadata_features": [
            "age",
            "sex",
        ],
        "fused_dimension": (fused_dimension),
        "learning_rate": float(best_trial["learning_rate"]),
        "weight_decay": float(best_trial["weight_decay"]),
        "class_weight": str(best_trial["class_weight"]),
        "fixed_epochs": int(best_trial["median_best_epoch"]),
        "selection_score": float(best_trial["selection_score"]),
        "macro_auroc_mean": float(best_trial["macro_auroc_mean"]),
        "balanced_accuracy_mean": float(best_trial["balanced_accuracy_mean"]),
        "folds": args.folds,
        "repeats": args.repeats,
        "total_splits": len(folds),
        "classifier_trial_budget": len(search_space),
        "pca_dimension_trial_budget": 1,
        "seed": args.seed,
        "selection_metric": ("mean((macro_auroc + balanced_accuracy) / 2) - 0.5 * SEM"),
        "validation_set_evaluated": (False),
        "test_set_evaluated": False,
        "total_elapsed_minutes": ((time.perf_counter() - run_start) / 60),
    }

    (args.output_directory / "selected_config.json").write_text(
        json.dumps(
            selected_config,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("=== PCA early-fusion train-CV results ===")

    print(
        trial_results[
            [
                "learning_rate",
                "weight_decay",
                "class_weight",
                "macro_auroc_mean",
                "macro_auroc_std",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
                "selection_score",
                "median_best_epoch",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== PCA explained variance ===")

    print(variance_summary.to_string(index=False))

    print()
    print("=== Selected configuration ===")

    print(
        json.dumps(
            selected_config,
            indent=2,
        )
    )

    print()
    print("Fixed validation was not evaluated.")
    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
