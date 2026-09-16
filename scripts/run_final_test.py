import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from lung_fusion_agent.baselines.data import (
    CLASS_NAMES,
    load_baseline_dataset,
)
from lung_fusion_agent.baselines.search import (
    SearchConfig,
    build_classifier,
)
from lung_fusion_agent.eval.metrics import (
    calculate_classification_metrics,
)
from lung_fusion_agent.fusion.late import (
    fuse_log_probabilities,
)

SEED = 42
BOOTSTRAP_REPLICATES = 2000
MAX_BOOTSTRAP_ATTEMPTS = 50000

BASELINE_NAMES = [
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
    "prism2_metadata",
]

FUSION_STREAM_NAMES = [
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
]

FUSION_NAME = "fused_fixed_logit"

FUSION_WEIGHTS = np.asarray(
    [
        0.05,
        0.25,
        0.05,
        0.65,
    ],
    dtype=np.float64,
)

BASELINE_CONFIG_PATH = Path("artifacts/results/baselines/selected_configs.json")

FINAL_STRATEGY_PATH = Path(
    "artifacts/results/fusion_finalists/selected_final_strategy.json"
)

OUTPUT_DIR = Path("artifacts/results/final_test")

COMPLETION_PATH = OUTPUT_DIR / "final_test_evaluation.json"


def load_locked_configs() -> dict[
    str,
    SearchConfig,
]:
    baseline_payload = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))

    if baseline_payload.get("test_set_evaluated") is not False:
        raise ValueError("Baseline file does not confirm that test was untouched.")

    final_payload = json.loads(FINAL_STRATEGY_PATH.read_text(encoding="utf-8"))

    if final_payload.get("selected_strategy") != "fixed_logit":
        raise ValueError("Locked final strategy is not fixed_logit.")

    if final_payload.get("test_set_evaluated") is not False:
        raise ValueError(
            "Final-strategy file does not confirm that test was untouched."
        )

    locked_weights = np.asarray(
        final_payload["fixed_logit_weights"],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        locked_weights,
        FUSION_WEIGHTS,
        rtol=0.0,
        atol=1e-12,
    )

    raw_configs = baseline_payload["selected_configs"]

    missing_configs = set(BASELINE_NAMES) - set(raw_configs)

    if missing_configs:
        raise ValueError(f"Missing locked configurations: {sorted(missing_configs)}")

    return {
        name: SearchConfig(
            c_value=float(raw_configs[name]["c_value"]),
            class_weight=(raw_configs[name]["class_weight"]),
        )
        for name in BASELINE_NAMES
    }


def fit_models(
    *,
    dataset: object,
    development_mask: np.ndarray,
    test_mask: np.ndarray,
    configs: dict[
        str,
        SearchConfig,
    ],
) -> dict[
    str,
    np.ndarray,
]:
    development_labels = dataset.labels[development_mask]

    predictions = {}

    for name in BASELINE_NAMES:
        print(
            "Fitting development model:",
            name,
        )

        features = dataset.representations[name]

        model = build_classifier(
            configs[name],
            seed=SEED,
        )

        model.fit(
            features[development_mask],
            development_labels,
        )

        classifier = model.named_steps["classifier"]

        expected_classes = np.arange(len(CLASS_NAMES))

        if not np.array_equal(
            classifier.classes_,
            expected_classes,
        ):
            raise RuntimeError(f"{name}: unexpected class order.")

        probabilities = model.predict_proba(features[test_mask]).astype(np.float32)

        expected_shape = (
            int(test_mask.sum()),
            len(CLASS_NAMES),
        )

        if probabilities.shape != expected_shape:
            raise RuntimeError(
                f"{name}: unexpected probability shape {probabilities.shape}."
            )

        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"{name}: predictions contain NaN or infinity.")

        predictions[name] = probabilities

    fusion_inputs = np.stack(
        [predictions[name] for name in (FUSION_STREAM_NAMES)],
        axis=1,
    )

    predictions[FUSION_NAME] = fuse_log_probabilities(
        fusion_inputs,
        FUSION_WEIGHTS,
    )

    return predictions


def evaluate(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[
    dict[str, float],
    dict[str, float],
]:
    metrics = calculate_classification_metrics(
        y_true=labels,
        probabilities=(probabilities),
        class_names=CLASS_NAMES,
    )

    record = {
        "macro_auroc": (metrics.macro_auroc),
        "balanced_accuracy": (metrics.balanced_accuracy),
        "composite_score": ((metrics.macro_auroc + metrics.balanced_accuracy) / 2.0),
    }

    return (
        record,
        metrics.per_class_auroc,
    )


def generate_bootstrap_indices(
    *,
    grouping_ids: np.ndarray,
    labels: np.ndarray,
) -> list[np.ndarray]:
    unique_groups = np.unique(grouping_ids)

    group_to_indices = {
        group: np.flatnonzero(grouping_ids == group) for group in unique_groups
    }

    rng = np.random.default_rng(SEED)

    samples = []
    attempts = 0

    while len(samples) < BOOTSTRAP_REPLICATES and attempts < MAX_BOOTSTRAP_ATTEMPTS:
        attempts += 1

        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )

        indices = np.concatenate(
            [group_to_indices[group] for group in (sampled_groups)]
        )

        # Macro AUROC requires every class
        # to be represented.
        if len(np.unique(labels[indices])) != len(CLASS_NAMES):
            continue

        samples.append(indices)

    if len(samples) != BOOTSTRAP_REPLICATES:
        raise RuntimeError("Could not generate enough valid bootstrap samples.")

    print(
        "Bootstrap samples:",
        len(samples),
    )

    print(
        "Bootstrap attempts:",
        attempts,
    )

    return samples


def run_bootstrap(
    *,
    labels: np.ndarray,
    predictions: dict[
        str,
        np.ndarray,
    ],
    bootstrap_indices: list[np.ndarray],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    bootstrap_records = []
    difference_records = []

    comparison_names = [name for name in predictions if name != FUSION_NAME]

    for replicate_index, indices in enumerate(bootstrap_indices):
        replicate_metrics = {}

        for name, probabilities in predictions.items():
            record, _ = evaluate(
                labels=labels[indices],
                probabilities=(probabilities[indices]),
            )

            replicate_metrics[name] = record

            bootstrap_records.append(
                {
                    "replicate_index": (replicate_index),
                    "configuration": (name),
                    **record,
                }
            )

        fused = replicate_metrics[FUSION_NAME]

        for comparison_name in comparison_names:
            comparison = replicate_metrics[comparison_name]

            difference_records.append(
                {
                    "replicate_index": (replicate_index),
                    "comparison": (comparison_name),
                    "macro_auroc_difference": (
                        fused["macro_auroc"] - comparison["macro_auroc"]
                    ),
                    "balanced_accuracy_difference": (
                        fused["balanced_accuracy"] - comparison["balanced_accuracy"]
                    ),
                    "composite_difference": (
                        fused["composite_score"] - comparison["composite_score"]
                    ),
                }
            )

    return (
        pd.DataFrame(bootstrap_records),
        pd.DataFrame(difference_records),
    )


def summarize_intervals(
    bootstrap_frame: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for name, frame in bootstrap_frame.groupby(
        "configuration",
        sort=False,
    ):
        records.append(
            {
                "configuration": name,
                "macro_auroc_ci_lower": (frame["macro_auroc"].quantile(0.025)),
                "macro_auroc_ci_upper": (frame["macro_auroc"].quantile(0.975)),
                "balanced_accuracy_ci_lower": (
                    frame["balanced_accuracy"].quantile(0.025)
                ),
                "balanced_accuracy_ci_upper": (
                    frame["balanced_accuracy"].quantile(0.975)
                ),
            }
        )

    return pd.DataFrame(records)


def summarize_differences(
    difference_frame: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    metric_names = [
        "macro_auroc",
        "balanced_accuracy",
        "composite",
    ]

    for comparison_name, frame in difference_frame.groupby(
        "comparison",
        sort=False,
    ):
        record = {"comparison": (comparison_name)}

        for metric_name in metric_names:
            column = f"{metric_name}_difference"

            record[f"{metric_name}_difference_mean"] = frame[column].mean()

            record[f"{metric_name}_difference_ci_lower"] = frame[column].quantile(0.025)

            record[f"{metric_name}_difference_ci_upper"] = frame[column].quantile(0.975)

            record[f"{metric_name}_probability_fused_greater"] = (
                frame[column] > 0
            ).mean()

        records.append(record)

    return pd.DataFrame(records)


def save_confusion_matrices(
    *,
    labels: np.ndarray,
    predictions: dict[
        str,
        np.ndarray,
    ],
) -> None:
    output_directory = OUTPUT_DIR / "confusion_matrices"

    output_directory.mkdir(
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

        frame.to_csv(output_directory / f"{name}.csv")


def save_test_predictions(
    *,
    dataset: object,
    test_mask: np.ndarray,
    predictions: dict[
        str,
        np.ndarray,
    ],
) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": (dataset.patient_ids[test_mask]),
            "grouping_id": (dataset.grouping_ids[test_mask]),
            "wsi_id": (dataset.wsi_ids[test_mask]),
            "true_label": (dataset.label_names[test_mask]),
        }
    )

    for name, probabilities in predictions.items():
        frame[f"{name}_prediction"] = [
            CLASS_NAMES[class_index] for class_index in (probabilities.argmax(axis=1))
        ]

        for class_index, class_name in enumerate(CLASS_NAMES):
            safe_class_name = class_name.lower().replace(
                " ",
                "_",
            )

            frame[f"{name}_probability_{safe_class_name}"] = probabilities[
                :,
                class_index,
            ]

    frame.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )


def main() -> None:
    if COMPLETION_PATH.exists():
        raise RuntimeError(
            "Final test evaluation has already "
            "completed. Existing marker: "
            f"{COMPLETION_PATH}"
        )

    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(
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
        raise ValueError("Expected 162 development records.")

    if int(test_mask.sum()) != 42:
        raise ValueError("Expected 42 test records.")

    development_groups = set(dataset.grouping_ids[development_mask])

    test_groups = set(dataset.grouping_ids[test_mask])

    if development_groups & test_groups:
        raise ValueError("Grouping leakage detected.")

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
        "Locked fusion weights:",
        FUSION_WEIGHTS.tolist(),
    )

    print()

    predictions = fit_models(
        dataset=dataset,
        development_mask=(development_mask),
        test_mask=test_mask,
        configs=configs,
    )

    test_labels = dataset.labels[test_mask]

    test_grouping_ids = dataset.grouping_ids[test_mask].astype(str)

    result_records = []
    per_class_records = []

    for name, probabilities in predictions.items():
        record, per_class_auroc = evaluate(
            labels=test_labels,
            probabilities=(probabilities),
        )

        result_records.append(
            {
                "configuration": (name),
                **record,
            }
        )

        for class_name, auroc in per_class_auroc.items():
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
        OUTPUT_DIR / "test_results.csv",
        index=False,
    )

    per_class_frame.to_csv(
        OUTPUT_DIR / "per_class_auroc.csv",
        index=False,
    )

    bootstrap_frame.to_csv(
        OUTPUT_DIR / "bootstrap_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    difference_frame.to_csv(
        OUTPUT_DIR / "paired_bootstrap_differences.csv.gz",
        index=False,
        compression="gzip",
    )

    paired_summary.to_csv(
        OUTPUT_DIR / "paired_difference_summary.csv",
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

    single_names = [
        "uni2",
        "virchow2",
        "prism2",
    ]

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
    print("=== Final test results ===")

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
        "Final test evaluation completed and locked:",
        COMPLETION_PATH,
    )


if __name__ == "__main__":
    main()
