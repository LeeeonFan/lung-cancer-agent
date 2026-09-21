"""Run the locked PyTorch Linear final test evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_final_test import (
    evaluate,
    generate_bootstrap_indices,
    run_bootstrap,
    summarize_differences,
    summarize_intervals,
)
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from lung_fusion_agent.baselines.data import (
    CLASS_NAMES,
    load_baseline_dataset,
)
from lung_fusion_agent.fusion.late import (
    fuse_log_probabilities,
)
from lung_fusion_agent.training.linear_classifier import (
    LinearTrainingConfig,
    choose_device,
    predict_probabilities,
    train_linear_classifier,
)

SEED = 42
BOOTSTRAP_REPLICATES = 2000

BASELINE_NAMES = (
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
    "prism2_metadata",
)

FUSION_STREAM_NAMES = (
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
)

FUSION_NAME = "fused_fixed_logit"

FUSION_WEIGHTS = np.asarray(
    [
        0.05,
        0.15,
        0.10,
        0.70,
    ],
    dtype=np.float64,
)

LINEAR_CONFIG_ROOT = Path("artifacts/results/linear_cv")

CLASSIFIER_SELECTION_PATH = Path(
    "artifacts/results/classifier_comparison/selection_summary.json"
)

CLASS_SPECIFIC_SELECTION_PATH = Path(
    "artifacts/results/class_specific_fusion/selected_class_specific.json"
)

OUTPUT_DIRECTORY = Path("artifacts/results/pytorch_final_test")

COMPLETION_PATH = OUTPUT_DIRECTORY / "final_test_evaluation.json"


def load_json(
    path: Path,
) -> dict[str, object]:
    """Load a required JSON artifact."""

    if not path.exists():
        raise FileNotFoundError(f"Missing locked artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def load_locked_configs() -> dict[
    str,
    LinearTrainingConfig,
]:
    """Load and validate all locked PyTorch configs."""

    classifier_selection = load_json(CLASSIFIER_SELECTION_PATH)

    if classifier_selection.get("selected_classifier_family") != "linear":
        raise ValueError("The locked classifier family is not PyTorch Linear")

    if classifier_selection.get("validation_set_evaluated") is not False:
        raise ValueError("Classifier selection does not confirm untouched validation")

    if classifier_selection.get("test_set_evaluated") is not False:
        raise ValueError("Classifier selection does not confirm untouched test")

    fusion_selection = load_json(CLASS_SPECIFIC_SELECTION_PATH)

    if fusion_selection.get("selected_method") != "global_log_fusion":
        raise ValueError("The locked fusion method is not global_log_fusion")

    if fusion_selection.get("materially_improved") is not False:
        raise ValueError("Class-specific fusion was unexpectedly marked improved")

    if fusion_selection.get("validation_set_evaluated") is not False:
        raise ValueError("Fusion selection does not confirm untouched validation")

    if fusion_selection.get("test_set_evaluated") is not False:
        raise ValueError("Fusion selection does not confirm untouched test")

    raw_weight_mapping = fusion_selection["global_weights"]

    locked_weights = np.asarray(
        [raw_weight_mapping[stream_name] for stream_name in (FUSION_STREAM_NAMES)],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        locked_weights,
        FUSION_WEIGHTS,
        rtol=0.0,
        atol=1e-12,
    )

    configs: dict[
        str,
        LinearTrainingConfig,
    ] = {}

    for representation in BASELINE_NAMES:
        config_path = LINEAR_CONFIG_ROOT / representation / "selected_config.json"

        payload = load_json(config_path)

        if payload.get("representation") != representation:
            raise ValueError(f"{representation}: representation mismatch")

        if payload.get("validation_set_evaluated") is not False:
            raise ValueError(
                f"{representation}: config does not confirm untouched validation"
            )

        if payload.get("test_set_evaluated") is not False:
            raise ValueError(
                f"{representation}: config does not confirm untouched test"
            )

        if int(payload["trial_budget"]) != 12:
            raise ValueError(f"{representation}: expected a 12-trial budget")

        configs[representation] = LinearTrainingConfig(
            epochs=int(payload["fixed_epochs"]),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            class_weight=str(payload["class_weight"]),
            seed=SEED,
            early_stopping_patience=None,
            minimum_improvement=1e-3,
            restore_best_state=False,
        )

    return configs


def fit_pytorch_models(
    *,
    dataset: object,
    development_mask: np.ndarray,
    test_mask: np.ndarray,
    configs: dict[
        str,
        LinearTrainingConfig,
    ],
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, object]],
]:
    """Fit all locked PyTorch Linear models."""

    development_labels = dataset.labels[development_mask].astype(np.int64)

    device = choose_device()

    print(
        "PyTorch device:",
        device,
    )

    predictions: dict[
        str,
        np.ndarray,
    ] = {}

    training_records: list[dict[str, object]] = []

    for representation in BASELINE_NAMES:
        print()
        print(
            "Fitting PyTorch model:",
            representation,
        )

        features = dataset.representations[representation].astype(np.float32)

        scaler = StandardScaler()

        development_features = scaler.fit_transform(features[development_mask]).astype(
            np.float32
        )

        test_features = scaler.transform(features[test_mask]).astype(np.float32)

        config = configs[representation]

        # The epoch count was selected using
        # train-only repeated CV. Development
        # data are passed as monitoring data so
        # the test set is never inspected during
        # model fitting.
        result = train_linear_classifier(
            train_features=(development_features),
            train_labels=(development_labels),
            validation_features=(development_features),
            validation_labels=(development_labels),
            number_of_classes=len(CLASS_NAMES),
            config=config,
            device=device,
        )

        if len(result.history) != config.epochs:
            raise RuntimeError(
                f"{representation}: expected "
                f"{config.epochs} epochs, "
                f"completed "
                f"{len(result.history)}"
            )

        test_tensor = torch.from_numpy(test_features).to(device)

        probabilities = predict_probabilities(
            result.model,
            test_tensor,
        ).astype(np.float32)

        expected_shape = (
            int(test_mask.sum()),
            len(CLASS_NAMES),
        )

        if probabilities.shape != (expected_shape):
            raise RuntimeError(
                f"{representation}: unexpected probability shape {probabilities.shape}"
            )

        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"{representation}: non-finite probabilities")

        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=1e-5,
            atol=1e-6,
        )

        predictions[representation] = probabilities

        final_history = result.history[-1]

        training_records.append(
            {
                "configuration": (representation),
                "input_dimension": int(development_features.shape[1]),
                "development_records": int(development_mask.sum()),
                "test_records": int(test_mask.sum()),
                "learning_rate": (config.learning_rate),
                "weight_decay": (config.weight_decay),
                "class_weight": (config.class_weight),
                "fixed_epochs": (config.epochs),
                "final_development_loss": (float(final_history["training_loss"])),
                "seed": config.seed,
            }
        )

        print(
            "  epochs:",
            config.epochs,
        )
        print(
            "  learning rate:",
            config.learning_rate,
        )
        print(
            "  weight decay:",
            config.weight_decay,
        )
        print(
            "  class weight:",
            config.class_weight,
        )

    fusion_inputs = np.stack(
        [predictions[stream_name] for stream_name in (FUSION_STREAM_NAMES)],
        axis=1,
    )

    predictions[FUSION_NAME] = fuse_log_probabilities(
        fusion_inputs,
        FUSION_WEIGHTS,
    ).astype(np.float32)

    return (
        predictions,
        training_records,
    )


def save_confusion_matrices(
    *,
    labels: np.ndarray,
    predictions: dict[
        str,
        np.ndarray,
    ],
) -> None:
    """Save one confusion matrix per configuration."""

    directory = OUTPUT_DIRECTORY / "confusion_matrices"

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, probabilities in predictions.items():
        predicted_labels = probabilities.argmax(axis=1)

        matrix = confusion_matrix(
            labels,
            predicted_labels,
            labels=np.arange(len(CLASS_NAMES)),
        )

        frame = pd.DataFrame(
            matrix,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        )

        frame.index.name = "true_label"

        frame.columns.name = "predicted_label"

        frame.to_csv(directory / f"{name}.csv")


def save_test_predictions(
    *,
    dataset: object,
    test_mask: np.ndarray,
    predictions: dict[
        str,
        np.ndarray,
    ],
) -> None:
    """Save patient-level test predictions."""

    frame = pd.DataFrame(
        {
            "patient_id": (dataset.patient_ids[test_mask]),
            "grouping_id": (dataset.grouping_ids[test_mask]),
            "wsi_id": (dataset.wsi_ids[test_mask]),
            "true_label": (dataset.label_names[test_mask]),
        }
    )

    for name, probabilities in predictions.items():
        predicted_indices = probabilities.argmax(axis=1)

        frame[f"{name}_prediction"] = [
            CLASS_NAMES[class_index] for class_index in (predicted_indices)
        ]

        for (
            class_index,
            class_name,
        ) in enumerate(CLASS_NAMES):
            safe_class_name = class_name.lower().replace(
                " ",
                "_",
            )

            frame[f"{name}_probability_{safe_class_name}"] = probabilities[
                :,
                class_index,
            ]

    frame.to_csv(
        OUTPUT_DIRECTORY / "test_predictions.csv",
        index=False,
    )


def main() -> None:
    """Run one locked PyTorch test evaluation."""

    if COMPLETION_PATH.exists():
        raise RuntimeError(
            "PyTorch final test evaluation "
            "has already completed. "
            "Existing marker: "
            f"{COMPLETION_PATH}"
        )

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    configs = load_locked_configs()

    dataset = load_baseline_dataset()

    development_mask = np.isin(
        dataset.splits,
        [
            "train",
            "validation",
        ],
    )

    test_mask = dataset.mask_for_split("test")

    if int(development_mask.sum()) != 162:
        raise ValueError("Expected 162 development records")

    if int(test_mask.sum()) != 42:
        raise ValueError("Expected 42 test records")

    development_groups = set(dataset.grouping_ids[development_mask])

    test_groups = set(dataset.grouping_ids[test_mask])

    if development_groups & test_groups:
        raise ValueError("Grouping leakage detected")

    print(
        "Development records:",
        int(development_mask.sum()),
    )

    print(
        "Test records:",
        int(test_mask.sum()),
    )

    print(
        "Test grouping IDs:",
        len(test_groups),
    )

    print(
        "Locked PyTorch fusion weights:",
        FUSION_WEIGHTS.tolist(),
    )

    (
        predictions,
        training_records,
    ) = fit_pytorch_models(
        dataset=dataset,
        development_mask=(development_mask),
        test_mask=test_mask,
        configs=configs,
    )

    pd.DataFrame(training_records).to_csv(
        OUTPUT_DIRECTORY / "locked_training_configs.csv",
        index=False,
    )

    test_labels = dataset.labels[test_mask]

    test_grouping_ids = dataset.grouping_ids[test_mask].astype(str)

    result_records: list[dict[str, object]] = []

    per_class_records: list[dict[str, object]] = []

    for name, probabilities in predictions.items():
        (
            record,
            per_class_auroc,
        ) = evaluate(
            labels=test_labels,
            probabilities=probabilities,
        )

        result_records.append(
            {
                "configuration": (name),
                **record,
            }
        )

        for (
            class_name,
            auroc,
        ) in per_class_auroc.items():
            per_class_records.append(
                {
                    "configuration": (name),
                    "class_name": (class_name),
                    "auroc": auroc,
                }
            )

    results_frame = pd.DataFrame(result_records)

    per_class_frame = pd.DataFrame(per_class_records)

    bootstrap_indices = generate_bootstrap_indices(
        grouping_ids=(test_grouping_ids),
        labels=test_labels,
    )

    (
        bootstrap_frame,
        difference_frame,
    ) = run_bootstrap(
        labels=test_labels,
        predictions=predictions,
        bootstrap_indices=(bootstrap_indices),
    )

    interval_frame = summarize_intervals(bootstrap_frame)

    paired_summary = summarize_differences(difference_frame)

    results_with_intervals = results_frame.merge(
        interval_frame,
        on="configuration",
        how="left",
        validate="one_to_one",
    )

    results_with_intervals.to_csv(
        OUTPUT_DIRECTORY / "test_results.csv",
        index=False,
    )

    per_class_frame.to_csv(
        OUTPUT_DIRECTORY / "per_class_auroc.csv",
        index=False,
    )

    bootstrap_frame.to_csv(
        OUTPUT_DIRECTORY / "bootstrap_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    difference_frame.to_csv(
        OUTPUT_DIRECTORY / "paired_bootstrap_differences.csv.gz",
        index=False,
        compression="gzip",
    )

    paired_summary.to_csv(
        OUTPUT_DIRECTORY / "paired_difference_summary.csv",
        index=False,
    )

    save_confusion_matrices(
        labels=test_labels,
        predictions=predictions,
    )

    save_test_predictions(
        dataset=dataset,
        test_mask=test_mask,
        predictions=predictions,
    )

    indexed_results = results_with_intervals.set_index("configuration")

    fused_row = indexed_results.loc[FUSION_NAME]

    single_names = (
        "uni2",
        "virchow2",
        "prism2",
    )

    fused_beats_all_singles = all(
        (
            float(fused_row["macro_auroc"])
            > float(
                indexed_results.loc[
                    name,
                    "macro_auroc",
                ]
            )
        )
        and (
            float(fused_row["balanced_accuracy"])
            > float(
                indexed_results.loc[
                    name,
                    "balanced_accuracy",
                ]
            )
        )
        for name in single_names
    )

    completion_payload = {
        "seed": SEED,
        "classifier_family": ("PyTorch Linear"),
        "evaluation_context": (
            "Second-stage PyTorch migration after the original sklearn test evaluation"
        ),
        "development_records": int(development_mask.sum()),
        "test_records": int(test_mask.sum()),
        "test_grouping_ids": len(test_groups),
        "bootstrap_unit": ("grouping_id cluster"),
        "bootstrap_replicates": (BOOTSTRAP_REPLICATES),
        "selected_strategy": (FUSION_NAME),
        "fusion_weights": (FUSION_WEIGHTS.tolist()),
        "best_single_locked_before_test": ("prism2"),
        "fused_strictly_beats_all_single_models_on_both_primary_metrics": (
            fused_beats_all_singles
        ),
        "test_set_evaluated": True,
        "test_evaluation_completed_once": (True),
    }

    COMPLETION_PATH.write_text(
        json.dumps(
            completion_payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=== Final PyTorch test results ===")

    print(
        results_with_intervals[
            [
                "configuration",
                "macro_auroc",
                "macro_auroc_ci_lower",
                "macro_auroc_ci_upper",
                "balanced_accuracy",
                "balanced_accuracy_ci_lower",
                "balanced_accuracy_ci_upper",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== Paired bootstrap: fused minus comparator ===")

    print(paired_summary.to_string(index=False))

    print()
    print("Best single model locked before test: prism2")

    print(
        "Fused strictly beats every single model on both primary metrics:",
        fused_beats_all_singles,
    )

    print(
        "PyTorch final test evaluation completed and locked:",
        COMPLETION_PATH,
    )


if __name__ == "__main__":
    main()
