"""Select Linear-classifier settings using train-only repeated CV."""

from __future__ import annotations

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
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
    train_linear_classifier,
)

SPLIT_PATH = Path("artifacts/splits/patient_splits.csv")

EMBEDDING_DIRECTORY = Path("data/processed/slide_embeddings")

MODEL_FILES = {
    "uni2": "uni2_slide_embeddings.npz",
    "virchow2": ("virchow2_slide_embeddings.npz"),
    "prism2": ("prism2_slide_embeddings.npz"),
}

LEARNING_RATES = (
    1e-4,
    3e-4,
    1e-3,
)

WEIGHT_DECAYS = (
    1e-3,
    1e-2,
)

CLASS_WEIGHTS = (
    "none",
    "balanced",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--representation",
        choices=(
            "metadata",
            "uni2",
            "virchow2",
            "prism2",
            "prism2_metadata",
            "early_concat",
            "early_l2_concat",
        ),
        default="prism2",
    )

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
        "--output-root",
        type=Path,
        default=Path("artifacts/results/linear_cv"),
    )

    return parser.parse_args()


def load_embedding(
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    path = EMBEDDING_DIRECTORY / MODEL_FILES[model_name]

    with np.load(
        path,
        allow_pickle=False,
    ) as cache:
        patient_ids = cache["patient_ids"].astype(str)

        embeddings = cache["embeddings"].astype(np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"{model_name} embeddings must be two-dimensional")

    if len(patient_ids) != len(embeddings):
        raise ValueError(f"{model_name} patient IDs and embeddings differ")

    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError(f"{model_name} contains duplicate patient IDs")

    if not np.isfinite(embeddings).all():
        raise ValueError(f"{model_name} contains non-finite values")

    return patient_ids, embeddings


def align_embedding(
    patient_ids: np.ndarray,
    model_name: str,
) -> np.ndarray:
    (
        embedding_patient_ids,
        embeddings,
    ) = load_embedding(model_name)

    by_patient = dict(
        zip(
            embedding_patient_ids,
            embeddings,
            strict=True,
        )
    )

    missing = [patient_id for patient_id in patient_ids if patient_id not in by_patient]

    if missing:
        raise ValueError(f"{model_name} missing {len(missing)} patients: {missing[:5]}")

    return np.stack([by_patient[patient_id] for patient_id in patient_ids])


def metadata_features(
    table: pd.DataFrame,
) -> np.ndarray:
    sex = (
        table["sex"]
        .astype(str)
        .str.lower()
        .map(
            {
                "female": 0.0,
                "male": 1.0,
            }
        )
    )

    if sex.isna().any():
        unknown = sorted(
            table.loc[
                sex.isna(),
                "sex",
            ].unique()
        )

        raise ValueError(f"Unknown sex values: {unknown}")

    return np.column_stack(
        (
            table["age"].to_numpy(dtype=np.float32),
            sex.to_numpy(dtype=np.float32),
        )
    )


def l2_normalize_rows(
    features: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True,
    )

    return features / np.clip(
        norms,
        1e-12,
        None,
    )


def build_features(
    representation: str,
    table: pd.DataFrame,
) -> np.ndarray:
    patient_ids = table["patient_id"].astype(str).to_numpy()

    metadata = metadata_features(table)

    if representation == "metadata":
        return metadata

    if representation in MODEL_FILES:
        return align_embedding(
            patient_ids,
            representation,
        )

    if representation == "prism2_metadata":
        prism2 = align_embedding(
            patient_ids,
            "prism2",
        )

        return np.concatenate(
            (
                prism2,
                metadata,
            ),
            axis=1,
        )

    if representation in {
        "early_concat",
        "early_l2_concat",
    }:
        streams = [
            align_embedding(
                patient_ids,
                model_name,
            )
            for model_name in (
                "uni2",
                "virchow2",
                "prism2",
            )
        ]

        if representation == "early_l2_concat":
            streams = [l2_normalize_rows(stream) for stream in streams]

        return np.concatenate(
            (
                *streams,
                metadata,
            ),
            axis=1,
        )

    raise ValueError(f"Unknown representation: {representation}")


def sem(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0

    return float(values.std(ddof=1) / math.sqrt(len(values)))


def main() -> None:
    args = parse_args()

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

    # Only the 141 training records enter
    # hyperparameter and epoch selection.
    train_table = split_table.loc[split_table["split"].eq("train")].reset_index(
        drop=True
    )

    if len(train_table) != 141:
        raise ValueError("Expected exactly 141 training records")

    features = build_features(
        args.representation,
        train_table,
    )

    label_encoder = LabelEncoder()

    labels = label_encoder.fit_transform(train_table["label"].astype(str))

    number_of_classes = len(label_encoder.classes_)

    splitter = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.seed,
    )

    # Materialize once so every trial uses
    # exactly the same 15 folds.
    folds = list(
        splitter.split(
            features,
            labels,
        )
    )

    search_space = list(
        product(
            LEARNING_RATES,
            WEIGHT_DECAYS,
            CLASS_WEIGHTS,
        )
    )

    expected_trial_count = 12

    if len(search_space) != (expected_trial_count):
        raise RuntimeError("The search space must contain exactly 12 configurations")

    output_directory = args.output_root / args.representation

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = choose_device()

    print(f"Representation: {args.representation}")

    print(f"Device: {device}")

    print(f"Training patients: {len(train_table)}")

    print(f"Feature dimension: {features.shape[1]}")

    print(f"Classes: {label_encoder.classes_.tolist()}")

    print(f"CV splits: {len(folds)}")

    print(f"Configurations: {len(search_space)}")

    fold_records: list[dict[str, object]] = []

    run_start = time.perf_counter()

    for trial_index, parameters in enumerate(search_space):
        (
            learning_rate,
            weight_decay,
            class_weight,
        ) = parameters

        print()
        print("=" * 70)

        print(f"Trial {trial_index + 1}/{len(search_space)}")

        print(f"learning_rate={learning_rate}")

        print(f"weight_decay={weight_decay}")

        print(f"class_weight={class_weight}")

        for fold_index, (
            fold_train_indices,
            fold_validation_indices,
        ) in enumerate(folds):
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
                epochs=args.max_epochs,
                learning_rate=(learning_rate),
                weight_decay=weight_decay,
                class_weight=class_weight,
                seed=(args.seed + trial_index * 1000 + fold_index),
                early_stopping_patience=(args.patience),
                minimum_improvement=(args.minimum_improvement),
            )

            fold_start = time.perf_counter()

            result = train_linear_classifier(
                train_features=(fold_train_features),
                train_labels=(fold_train_labels),
                validation_features=(fold_validation_features),
                validation_labels=(fold_validation_labels),
                number_of_classes=(number_of_classes),
                config=config,
                device=device,
            )

            elapsed_seconds = time.perf_counter() - fold_start

            history = pd.DataFrame(result.history)

            best_row = history.loc[history["epoch"].eq(result.best_epoch)].iloc[0]

            repeat_index = fold_index // args.folds

            fold_in_repeat = fold_index % args.folds

            fold_record = {
                "trial_index": (trial_index),
                "repeat_index": (repeat_index),
                "fold_index": (fold_in_repeat),
                "global_fold_index": (fold_index),
                "learning_rate": (learning_rate),
                "weight_decay": (weight_decay),
                "class_weight": (class_weight),
                "train_records": len(fold_train_indices),
                "validation_records": len(fold_validation_indices),
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
                f"{fold_index + 1:02d}/"
                f"{len(folds)} | "
                f"epoch="
                f"{result.best_epoch:03d} | "
                f"AUROC="
                f"{fold_record['macro_auroc']:.4f} | "
                f"BA="
                f"{fold_record['balanced_accuracy']:.4f}"
            )

        # Save after every completed trial.
        pd.DataFrame(fold_records).to_csv(
            output_directory / "cv_folds.csv",
            index=False,
        )

    fold_results = pd.DataFrame(fold_records)

    trial_records: list[dict[str, object]] = []

    for trial_index, group in fold_results.groupby(
        "trial_index",
        sort=True,
    ):
        composite_sem = sem(group["composite"])

        selection_score = group["composite"].mean() - 0.5 * composite_sem

        trial_records.append(
            {
                "trial_index": (int(trial_index)),
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
        output_directory / "cv_trials.csv",
        index=False,
    )

    best_trial = trial_results.iloc[0]

    selected_config = {
        "representation": (args.representation),
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
        "trial_budget": len(search_space),
        "seed": args.seed,
        "selection_metric": ("mean((macro_auroc + balanced_accuracy) / 2) - 0.5 * SEM"),
        "validation_set_evaluated": (False),
        "test_set_evaluated": False,
        "total_elapsed_minutes": ((time.perf_counter() - run_start) / 60),
    }

    selected_path = output_directory / "selected_config.json"

    selected_path.write_text(
        json.dumps(
            selected_config,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 70)

    print("=== Linear train-CV results ===")

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
