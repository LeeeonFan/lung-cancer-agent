import json
from pathlib import Path

import numpy as np
import pandas as pd

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
from lung_fusion_agent.fusion.gated import (
    MetadataGatedLogitFusion,
)
from lung_fusion_agent.fusion.late import (
    fuse_log_probabilities,
)

SEED = 42

STREAM_NAMES = [
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
]

FIXED_LOGIT_WEIGHTS = np.asarray(
    [
        0.05,
        0.25,
        0.05,
        0.65,
    ],
    dtype=np.float64,
)

GATED_REGULARIZATION = 1.0
GATED_CLASS_WEIGHT_BALANCED = True

BASELINE_CONFIG_PATH = Path("artifacts/results/baselines/selected_configs.json")

OOF_CACHE_PATH = Path("artifacts/results/fusion_agent/oof_probabilities.npz")

OUTPUT_DIR = Path("artifacts/results/fusion_finalists")


def load_selected_configs() -> dict[
    str,
    SearchConfig,
]:
    payload = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))

    if payload.get("test_set_evaluated") is not False:
        raise ValueError("Baseline file does not confirm that test was untouched.")

    raw_configs = payload["selected_configs"]

    return {
        stream_name: SearchConfig(
            c_value=float(raw_configs[stream_name]["c_value"]),
            class_weight=(raw_configs[stream_name]["class_weight"]),
        )
        for stream_name in STREAM_NAMES
    }


def load_averaged_train_oof(
    *,
    expected_patient_ids: np.ndarray,
) -> np.ndarray:
    with np.load(
        OOF_CACHE_PATH,
        allow_pickle=False,
    ) as cache:
        patient_ids = cache["patient_ids"].astype(str)

        stream_names = cache["stream_names"].astype(str).tolist()

        class_names = cache["class_names"].astype(str).tolist()

        validation_masks = cache["validation_masks"].astype(bool)

        probabilities = cache["probabilities"].astype(np.float32)

    if not np.array_equal(
        patient_ids,
        expected_patient_ids,
    ):
        raise ValueError("OOF patient order does not match the train dataset order.")

    if stream_names != STREAM_NAMES:
        raise ValueError("OOF stream order mismatch.")

    if class_names != CLASS_NAMES:
        raise ValueError("OOF class order mismatch.")

    prediction_counts = validation_masks.sum(axis=0)

    if not np.all(prediction_counts == 3):
        raise ValueError("Each train patient must have exactly three OOF predictions.")

    probability_sums = np.nansum(
        probabilities,
        axis=0,
    )

    averaged_probabilities = (
        probability_sums
        / prediction_counts[
            :,
            None,
            None,
        ]
    )

    if not np.isfinite(averaged_probabilities).all():
        raise ValueError("Averaged OOF probabilities contain NaN or infinity.")

    row_sums = averaged_probabilities.sum(axis=2)

    np.testing.assert_allclose(
        row_sums,
        np.ones_like(row_sums),
        rtol=1e-5,
        atol=1e-5,
    )

    return averaged_probabilities.astype(np.float32)


def fit_base_models_and_predict(
    *,
    dataset: object,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    selected_configs: dict[
        str,
        SearchConfig,
    ],
) -> np.ndarray:
    train_labels = dataset.labels[train_mask]

    validation_parts = []

    for stream_name in STREAM_NAMES:
        print(
            "Fitting full-train base model:",
            stream_name,
        )

        features = dataset.representations[stream_name]

        model = build_classifier(
            selected_configs[stream_name],
            seed=SEED,
        )

        model.fit(
            features[train_mask],
            train_labels,
        )

        classifier = model.named_steps["classifier"]

        expected_classes = np.arange(len(CLASS_NAMES))

        if not np.array_equal(
            classifier.classes_,
            expected_classes,
        ):
            raise RuntimeError(f"{stream_name}: unexpected classifier class order.")

        probabilities = model.predict_proba(features[validation_mask]).astype(
            np.float32
        )

        validation_parts.append(probabilities)

    stacked_probabilities = np.stack(
        validation_parts,
        axis=1,
    )

    expected_shape = (
        int(validation_mask.sum()),
        len(STREAM_NAMES),
        len(CLASS_NAMES),
    )

    if stacked_probabilities.shape != expected_shape:
        raise RuntimeError(
            "Unexpected validation probability "
            f"shape: {stacked_probabilities.shape}; "
            f"expected {expected_shape}."
        )

    return stacked_probabilities


def metric_record(
    *,
    strategy: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    metrics = calculate_classification_metrics(
        y_true=labels,
        probabilities=(probabilities),
        class_names=CLASS_NAMES,
    )

    composite_score = (metrics.macro_auroc + metrics.balanced_accuracy) / 2.0

    return {
        "strategy": strategy,
        "macro_auroc": (metrics.macro_auroc),
        "balanced_accuracy": (metrics.balanced_accuracy),
        "composite_score": (composite_score),
        "per_class_auroc_json": (
            json.dumps(
                metrics.per_class_auroc,
                sort_keys=True,
            )
        ),
    }


def choose_final_strategy(
    results: pd.DataFrame,
) -> tuple[
    str,
    str,
]:
    fixed = results.set_index("strategy").loc["fixed_logit"]

    gated = results.set_index("strategy").loc["metadata_gated"]

    composite_gain = gated["composite_score"] - fixed["composite_score"]

    auroc_change = gated["macro_auroc"] - fixed["macro_auroc"]

    balanced_accuracy_change = gated["balanced_accuracy"] - fixed["balanced_accuracy"]

    # Predeclared variance-aware rule:
    # choose the more complex gated model only
    # if it produces a meaningful validation gain
    # without materially degrading either primary metric.
    choose_gated = (
        composite_gain >= 0.005
        and auroc_change >= -0.005
        and balanced_accuracy_change >= -0.01
    )

    if choose_gated:
        return (
            "metadata_gated",
            (
                "Gated fusion improved validation "
                "composite score by at least 0.005 "
                "without materially degrading either "
                "primary metric."
            ),
        )

    return (
        "fixed_logit",
        (
            "Gated fusion did not clear the "
            "predeclared minimum validation gain; "
            "select the simpler fixed logit fusion."
        ),
    )


def main() -> None:
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_baseline_dataset()

    train_mask = dataset.mask_for_split("train")

    validation_mask = dataset.mask_for_split("validation")

    train_patient_ids = dataset.patient_ids[train_mask].astype(str)

    train_labels = dataset.labels[train_mask]

    validation_labels = dataset.labels[validation_mask]

    train_oof_probabilities = load_averaged_train_oof(
        expected_patient_ids=(train_patient_ids)
    )

    selected_configs = load_selected_configs()

    validation_base_probabilities = fit_base_models_and_predict(
        dataset=dataset,
        train_mask=train_mask,
        validation_mask=(validation_mask),
        selected_configs=(selected_configs),
    )

    fixed_probabilities = fuse_log_probabilities(
        validation_base_probabilities,
        FIXED_LOGIT_WEIGHTS,
    )

    gated_model = MetadataGatedLogitFusion(
        prior_weights=(FIXED_LOGIT_WEIGHTS),
        regularization=(GATED_REGULARIZATION),
        class_weight_balanced=(GATED_CLASS_WEIGHT_BALANCED),
    )

    gated_model.fit(
        metadata=(dataset.representations["metadata"][train_mask]),
        probabilities=(train_oof_probabilities),
        labels=train_labels,
    )

    gated_probabilities = gated_model.predict_proba(
        metadata=(dataset.representations["metadata"][validation_mask]),
        probabilities=(validation_base_probabilities),
    )

    validation_gates = gated_model.predict_gates(
        dataset.representations["metadata"][validation_mask]
    )

    results = pd.DataFrame(
        [
            metric_record(
                strategy=("fixed_logit"),
                labels=(validation_labels),
                probabilities=(fixed_probabilities),
            ),
            metric_record(
                strategy=("metadata_gated"),
                labels=(validation_labels),
                probabilities=(gated_probabilities),
            ),
        ]
    )

    (
        selected_strategy,
        selection_reason,
    ) = choose_final_strategy(results)

    results.to_csv(
        OUTPUT_DIR / "validation_results.csv",
        index=False,
    )

    prediction_frame = pd.DataFrame(
        {
            "patient_id": (dataset.patient_ids[validation_mask]),
            "wsi_id": (dataset.wsi_ids[validation_mask]),
            "true_label": (dataset.label_names[validation_mask]),
            "fixed_prediction": [
                CLASS_NAMES[index] for index in (fixed_probabilities.argmax(axis=1))
            ],
            "gated_prediction": [
                CLASS_NAMES[index] for index in (gated_probabilities.argmax(axis=1))
            ],
        }
    )

    for class_index, class_name in enumerate(CLASS_NAMES):
        safe_name = class_name.lower().replace(
            " ",
            "_",
        )

        prediction_frame[f"fixed_probability_{safe_name}"] = fixed_probabilities[
            :,
            class_index,
        ]

        prediction_frame[f"gated_probability_{safe_name}"] = gated_probabilities[
            :,
            class_index,
        ]

    prediction_frame.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    selected_payload = {
        "selected_strategy": (selected_strategy),
        "selection_reason": (selection_reason),
        "selection_rule": {
            "minimum_composite_gain_for_gated": (0.005),
            "maximum_allowed_auroc_decrease": (0.005),
            "maximum_allowed_balanced_accuracy_decrease": (0.01),
        },
        "fixed_logit_weights": (FIXED_LOGIT_WEIGHTS.tolist()),
        "gated_configuration": {
            "regularization": (GATED_REGULARIZATION),
            "class_weight_balanced": (GATED_CLASS_WEIGHT_BALANCED),
            "optimization_success": (gated_model.optimization_success_),
            "optimization_message": (gated_model.optimization_message_),
            "mean_validation_gates": {
                stream_name: float(
                    validation_gates[
                        :,
                        stream_index,
                    ].mean()
                )
                for (
                    stream_index,
                    stream_name,
                ) in enumerate(STREAM_NAMES)
            },
        },
        "validation_patient_count": int(validation_mask.sum()),
        "validation_evaluated_once": True,
        "test_set_evaluated": False,
    }

    (OUTPUT_DIR / "selected_final_strategy.json").write_text(
        json.dumps(
            selected_payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=== Fusion finalist validation ===")

    print(
        results[
            [
                "strategy",
                "macro_auroc",
                "balanced_accuracy",
                "composite_score",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Selected final strategy:",
        selected_strategy,
    )

    print(
        "Reason:",
        selection_reason,
    )

    print()
    print(
        "Gated optimization success:",
        gated_model.optimization_success_,
    )

    print("Mean validation gates:")

    for (
        stream_index,
        stream_name,
    ) in enumerate(STREAM_NAMES):
        print(f"  {stream_name}: {validation_gates[:, stream_index].mean():.6f}")

    print()
    print("Test set was not evaluated.")
    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
