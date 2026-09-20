"""Evaluate train-CV-selected Linear models on fixed validation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from run_linear_training_curves import (
    build_features,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
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

OUTPUT_ROOT = Path("artifacts/results/linear_validation")

REPRESENTATIONS = (
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
    "prism2_metadata",
    "early_concat",
    "early_l2_concat",
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

    validation_loss = log_loss(
        labels,
        probabilities,
        labels=np.arange(number_of_classes),
    )

    composite = (macro_auroc + balanced_accuracy) / 2

    return {
        "macro_auroc": float(macro_auroc),
        "balanced_accuracy": float(balanced_accuracy),
        "validation_loss": float(validation_loss),
        "composite": float(composite),
    }


def plot_history(
    history: pd.DataFrame,
    number_of_classes: int,
    locked_epoch: int,
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
            marker="o",
            markersize=3,
        )

        axis.axvline(
            locked_epoch,
            color="tab:purple",
            linestyle=":",
            linewidth=1.5,
            label=(f"CV-locked epoch = {locked_epoch}"),
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
    split_table = pd.read_csv(SPLIT_PATH)

    split_table["patient_id"] = split_table["patient_id"].astype(str)

    split_table["split"] = split_table["split"].astype(str).str.lower()

    # Test rows are discarded before
    # features are constructed.
    development_table = split_table.loc[
        split_table["split"].isin(
            (
                "train",
                "validation",
            )
        )
    ].reset_index(drop=True)

    split_counts = development_table["split"].value_counts().to_dict()

    if split_counts != {
        "train": 141,
        "validation": 21,
    }:
        raise ValueError(f"Unexpected development counts: {split_counts}")

    train_mask = development_table["split"].eq("train").to_numpy()

    validation_mask = development_table["split"].eq("validation").to_numpy()

    label_encoder = LabelEncoder()

    labels = label_encoder.fit_transform(development_table["label"].astype(str))

    number_of_classes = len(label_encoder.classes_)

    device = choose_device()

    print(f"Device: {device}")
    print("Test rows were removed before feature construction.")

    validation_results = []

    for representation in REPRESENTATIONS:
        print()
        print("=" * 70)
        print(f"Representation: {representation}")

        selected_path = CONFIG_ROOT / representation / "selected_config.json"

        if not selected_path.exists():
            raise FileNotFoundError(selected_path)

        selected = json.loads(selected_path.read_text())

        features = build_features(
            representation,
            development_table,
        )

        scaler = StandardScaler()

        train_features = scaler.fit_transform(features[train_mask]).astype(np.float32)

        validation_features = scaler.transform(features[validation_mask]).astype(
            np.float32
        )

        train_labels = labels[train_mask]

        validation_labels = labels[validation_mask]

        fixed_epochs = int(selected["fixed_epochs"])

        config = LinearTrainingConfig(
            epochs=fixed_epochs,
            learning_rate=float(selected["learning_rate"]),
            weight_decay=float(selected["weight_decay"]),
            class_weight=str(selected["class_weight"]),
            seed=int(selected["seed"]),
            early_stopping_patience=None,
            minimum_improvement=1e-3,
            restore_best_state=False,
        )

        result = train_linear_classifier(
            train_features=(train_features),
            train_labels=train_labels,
            validation_features=(validation_features),
            validation_labels=(validation_labels),
            number_of_classes=(number_of_classes),
            config=config,
            device=device,
        )

        validation_tensor = torch.from_numpy(validation_features).to(device)

        probabilities = predict_probabilities(
            result.model,
            validation_tensor,
        )

        metrics = calculate_metrics(
            validation_labels,
            probabilities,
            number_of_classes,
        )

        predictions = probabilities.argmax(axis=1)

        output_directory = OUTPUT_ROOT / representation

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        history = pd.DataFrame(result.history)

        history.to_csv(
            output_directory / "training_history.csv",
            index=False,
        )

        plot_history(
            history,
            number_of_classes=(number_of_classes),
            locked_epoch=(fixed_epochs),
            output_path=(output_directory / "training_curves.png"),
        )

        prediction_table = pd.DataFrame(
            {
                "patient_id": (
                    development_table.loc[
                        validation_mask,
                        "patient_id",
                    ].to_numpy()
                ),
                "true_label": (label_encoder.inverse_transform(validation_labels)),
                "predicted_label": (label_encoder.inverse_transform(predictions)),
            }
        )

        for class_index, class_name in enumerate(label_encoder.classes_):
            prediction_table[f"probability_{class_name}"] = probabilities[
                :,
                class_index,
            ]

        prediction_table.to_csv(
            output_directory / "validation_predictions.csv",
            index=False,
        )

        checkpoint = result.checkpoint()

        checkpoint.update(
            {
                "representation": (representation),
                "checkpoint_epoch": (fixed_epochs),
                "selection_source": ("train-only repeated CV"),
                "classes": (label_encoder.classes_.tolist()),
                "scaler_mean": (scaler.mean_),
                "scaler_scale": (scaler.scale_),
            }
        )

        torch.save(
            checkpoint,
            output_directory / "final_checkpoint.pt",
        )

        run_metadata = {
            "representation": (representation),
            "test_set_evaluated": False,
            "configuration_source": (str(selected_path)),
            "train_records": int(train_mask.sum()),
            "validation_records": int(validation_mask.sum()),
            "input_dimension": int(train_features.shape[1]),
            "device": str(device),
            "config": asdict(config),
            "validation_metrics": (metrics),
        }

        (output_directory / "run_metadata.json").write_text(
            json.dumps(
                run_metadata,
                indent=2,
            )
            + "\n"
        )

        validation_results.append(
            {
                "representation": (representation),
                "learning_rate": (config.learning_rate),
                "weight_decay": (config.weight_decay),
                "class_weight": (config.class_weight),
                "fixed_epochs": (fixed_epochs),
                **metrics,
            }
        )

        print(f"AUROC: {metrics['macro_auroc']:.6f}")

        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.6f}")

        print(f"Validation loss: {metrics['validation_loss']:.6f}")

    results = pd.DataFrame(validation_results).sort_values(
        "composite",
        ascending=False,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_ROOT / "validation_results.csv",
        index=False,
    )

    summary = {
        "test_set_evaluated": False,
        "configuration_selection": ("train-only repeated 5-fold CV x 3 repeats"),
        "representations": list(REPRESENTATIONS),
        "results": (results.to_dict(orient="records")),
    }

    (OUTPUT_ROOT / "validation_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=== Fixed validation results ===")

    print(results.to_string(index=False))

    print()
    print("Test set was not loaded or evaluated.")


if __name__ == "__main__":
    main()
