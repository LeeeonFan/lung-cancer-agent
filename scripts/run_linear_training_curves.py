"""Train and plot an epoch-based linear classifier.

Only the fixed train and validation splits are used.
Test rows are removed before features are constructed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    "uni2": ("uni2_slide_embeddings.npz"),
    "virchow2": ("virchow2_slide_embeddings.npz"),
    "prism2": ("prism2_slide_embeddings.npz"),
}


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
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--class-weight",
        choices=(
            "none",
            "balanced",
        ),
        default="none",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--minimum-improvement",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/results/linear_training"),
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
        raise ValueError(f"{model_name} contains non-finite embeddings")

    return patient_ids, embeddings


def align_embedding(
    split_patient_ids: np.ndarray,
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

    missing = [
        patient_id for patient_id in split_patient_ids if patient_id not in by_patient
    ]

    if missing:
        raise ValueError(
            f"{model_name} is missing {len(missing)} patients: {missing[:5]}"
        )

    return np.stack([by_patient[patient_id] for patient_id in split_patient_ids])


def metadata_features(
    split_table: pd.DataFrame,
) -> np.ndarray:
    sex = (
        split_table["sex"]
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
            split_table.loc[
                sex.isna(),
                "sex",
            ].unique()
        )

        raise ValueError(f"Unknown sex values: {unknown}")

    return np.column_stack(
        (
            split_table["age"].to_numpy(dtype=np.float32),
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
    split_table: pd.DataFrame,
) -> np.ndarray:
    patient_ids = split_table["patient_id"].astype(str).to_numpy()

    metadata = metadata_features(split_table)

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


def plot_history(
    history: pd.DataFrame,
    number_of_classes: int,
    best_epoch: int,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )

    panels = (
        (
            "training_loss",
            "Training loss",
            "Cross-entropy",
            "tab:blue",
        ),
        (
            "validation_loss",
            "Validation loss",
            "Cross-entropy",
            "tab:orange",
        ),
        (
            "validation_macro_auroc",
            "Validation macro AUROC",
            "Macro AUROC",
            "tab:green",
        ),
        (
            "validation_balanced_accuracy",
            "Validation balanced accuracy",
            "Balanced accuracy",
            "tab:red",
        ),
    )

    for axis, panel in zip(
        axes.flat,
        panels,
        strict=True,
    ):
        (
            column,
            title,
            ylabel,
            color,
        ) = panel

        axis.plot(
            history["epoch"],
            history[column],
            color=color,
        )

        axis.axvline(
            best_epoch,
            color="tab:purple",
            linestyle=":",
            linewidth=1.5,
            label=(f"best epoch = {best_epoch}"),
        )

        axis.set(
            title=title,
            xlabel="Epoch",
            ylabel=ylabel,
        )

        axis.grid(alpha=0.25)
        axis.legend()

    axes[1, 0].axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1,
        label="random = 0.5",
    )

    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    random_ba = 1 / number_of_classes

    axes[1, 1].axhline(
        random_ba,
        color="black",
        linestyle="--",
        linewidth=1,
        label=(f"random = {random_ba:.3f}"),
    )

    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


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
        raise ValueError(f"Split table is missing: {sorted(missing_columns)}")

    split_table["patient_id"] = split_table["patient_id"].astype(str)

    split_table["split"] = split_table["split"].astype(str).str.lower()

    if split_table["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs in patient_splits.csv")

    # Remove test rows before building
    # classifier inputs.
    development_table = split_table.loc[
        split_table["split"].isin(
            (
                "train",
                "validation",
            )
        )
    ].reset_index(drop=True)

    split_counts = development_table["split"].value_counts().to_dict()

    expected_counts = {
        "train": 141,
        "validation": 21,
    }

    if split_counts != expected_counts:
        raise ValueError(f"Unexpected development split counts: {split_counts}")

    features = build_features(
        args.representation,
        development_table,
    )

    label_encoder = LabelEncoder()

    labels = label_encoder.fit_transform(development_table["label"].astype(str))

    train_mask = development_table["split"].eq("train").to_numpy()

    validation_mask = development_table["split"].eq("validation").to_numpy()

    scaler = StandardScaler()

    train_features = scaler.fit_transform(features[train_mask]).astype(np.float32)

    validation_features = scaler.transform(features[validation_mask]).astype(np.float32)

    train_labels = labels[train_mask]

    validation_labels = labels[validation_mask]

    config = LinearTrainingConfig(
        epochs=args.epochs,
        learning_rate=(args.learning_rate),
        weight_decay=args.weight_decay,
        class_weight=args.class_weight,
        seed=args.seed,
        early_stopping_patience=(args.early_stopping_patience),
        minimum_improvement=(args.minimum_improvement),
    )

    device = choose_device()

    print(f"Device: {device}")

    print(
        "Representation:",
        args.representation,
    )

    print(
        "Train shape:",
        train_features.shape,
    )

    print(
        "Validation shape:",
        validation_features.shape,
    )

    print(
        "Classes:",
        label_encoder.classes_.tolist(),
    )

    result = train_linear_classifier(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=(validation_features),
        validation_labels=(validation_labels),
        number_of_classes=len(label_encoder.classes_),
        config=config,
        device=device,
    )

    output_directory = args.output_root / args.representation

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = pd.DataFrame(result.history)

    history_path = output_directory / "training_history.csv"

    plot_path = output_directory / "training_curves.png"

    checkpoint_path = output_directory / "best_checkpoint.pt"

    metadata_path = output_directory / "run_metadata.json"

    history.to_csv(
        history_path,
        index=False,
    )

    plot_history(
        history,
        number_of_classes=len(label_encoder.classes_),
        best_epoch=result.best_epoch,
        output_path=plot_path,
    )

    checkpoint = result.checkpoint()

    checkpoint.update(
        {
            "representation": (args.representation),
            "classes": (label_encoder.classes_.tolist()),
            "scaler_mean": (scaler.mean_),
            "scaler_scale": (scaler.scale_),
        }
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    best_row = history.loc[history["epoch"].eq(result.best_epoch)].iloc[0]

    run_metadata = {
        "diagnostic_only": True,
        "test_set_evaluated": False,
        "representation": (args.representation),
        "device": str(device),
        "input_dimension": int(train_features.shape[1]),
        "classes": (label_encoder.classes_.tolist()),
        "train_records": int(train_mask.sum()),
        "validation_records": int(validation_mask.sum()),
        "config": asdict(config),
        "epochs_completed": len(history),
        "best_epoch_by_validation_composite": (result.best_epoch),
        "best_validation_loss": float(best_row["validation_loss"]),
        "best_validation_macro_auroc": (float(best_row["validation_macro_auroc"])),
        "best_validation_balanced_accuracy": (
            float(best_row["validation_balanced_accuracy"])
        ),
        "best_validation_composite": (result.best_validation_composite),
    }

    metadata_path.write_text(
        json.dumps(
            run_metadata,
            indent=2,
        )
        + "\n"
    )

    for _, row in history.iterrows():
        epoch = int(row["epoch"])

        if epoch == 1 or epoch % 10 == 0 or epoch == len(history):
            print(
                f"epoch={epoch:03d} "
                f"train_loss="
                f"{row['training_loss']:.4f} "
                f"val_loss="
                f"{row['validation_loss']:.4f} "
                f"val_auroc="
                f"{row['validation_macro_auroc']:.4f} "
                f"val_ba="
                f"{row['validation_balanced_accuracy']:.4f}"
            )

    print(f"Best epoch: {result.best_epoch}")

    print(f"History: {history_path}")
    print(f"Plot: {plot_path}")

    print(f"Checkpoint: {checkpoint_path}")

    print(f"Metadata: {metadata_path}")

    print("Test set was not loaded or evaluated.")


if __name__ == "__main__":
    main()
