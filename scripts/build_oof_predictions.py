import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)

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

SEED = 42
N_SPLITS = 5
N_REPEATS = 3

BASELINE_CONFIG_PATH = Path("artifacts/results/baselines/selected_configs.json")

OUTPUT_DIR = Path("artifacts/results/fusion_agent")

STREAM_NAMES = [
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
]


def load_selected_configs() -> dict[
    str,
    SearchConfig,
]:
    payload = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))

    if payload.get("test_set_evaluated") is not False:
        raise ValueError(
            "Baseline configuration file does not confirm that test was untouched."
        )

    raw_configs = payload["selected_configs"]

    configs = {}

    for stream_name in STREAM_NAMES:
        raw_config = raw_configs[stream_name]

        configs[stream_name] = SearchConfig(
            c_value=float(raw_config["c_value"]),
            class_weight=(raw_config["class_weight"]),
        )

    return configs


def main() -> None:
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_baseline_dataset()
    selected_configs = load_selected_configs()

    train_mask = dataset.mask_for_split("train")

    train_patient_ids = dataset.patient_ids[train_mask].astype(str)

    train_grouping_ids = dataset.grouping_ids[train_mask].astype(str)

    train_labels = dataset.labels[train_mask]

    if len(np.unique(train_grouping_ids)) != len(train_grouping_ids):
        raise ValueError(
            "Train contains repeated grouping IDs. "
            "A group-aware CV splitter is required."
        )

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )

    folds = list(
        splitter.split(
            np.zeros(len(train_labels)),
            train_labels,
        )
    )

    fold_count = N_SPLITS * N_REPEATS

    if len(folds) != fold_count:
        raise RuntimeError("Unexpected number of CV folds.")

    probability_cache = np.full(
        (
            fold_count,
            len(train_labels),
            len(STREAM_NAMES),
            len(CLASS_NAMES),
        ),
        np.nan,
        dtype=np.float32,
    )

    validation_masks = np.zeros(
        (
            fold_count,
            len(train_labels),
        ),
        dtype=bool,
    )

    metric_records = []
    assignment_records = []

    for split_index, (
        fit_indices,
        validation_indices,
    ) in enumerate(folds):
        repeat_index = split_index // N_SPLITS
        fold_index = split_index % N_SPLITS

        validation_masks[
            split_index,
            validation_indices,
        ] = True

        for patient_index in validation_indices:
            assignment_records.append(
                {
                    "split_index": split_index,
                    "repeat_index": (repeat_index),
                    "fold_index": fold_index,
                    "patient_index": int(patient_index),
                    "patient_id": (train_patient_ids[patient_index]),
                    "label_index": int(train_labels[patient_index]),
                    "label": CLASS_NAMES[train_labels[patient_index]],
                }
            )

        print()
        print(
            f"Fold {split_index + 1}/"
            f"{fold_count} "
            f"(repeat={repeat_index}, "
            f"fold={fold_index})"
        )

        for stream_index, stream_name in enumerate(STREAM_NAMES):
            features = dataset.representations[stream_name][train_mask]

            config = selected_configs[stream_name]

            model = build_classifier(
                config,
                seed=SEED,
            )

            model.fit(
                features[fit_indices],
                train_labels[fit_indices],
            )

            classifier = model.named_steps["classifier"]

            expected_classes = np.arange(len(CLASS_NAMES))

            if not np.array_equal(
                classifier.classes_,
                expected_classes,
            ):
                raise RuntimeError(f"{stream_name}: unexpected class order.")

            probabilities = model.predict_proba(features[validation_indices]).astype(
                np.float32
            )

            probability_cache[
                split_index,
                validation_indices,
                stream_index,
                :,
            ] = probabilities

            metrics = calculate_classification_metrics(
                y_true=train_labels[validation_indices],
                probabilities=(probabilities),
                class_names=(CLASS_NAMES),
            )

            metric_records.append(
                {
                    "split_index": (split_index),
                    "repeat_index": (repeat_index),
                    "fold_index": fold_index,
                    "stream": stream_name,
                    "macro_auroc": (metrics.macro_auroc),
                    "balanced_accuracy": (metrics.balanced_accuracy),
                    "composite_score": (
                        (metrics.macro_auroc + metrics.balanced_accuracy) / 2.0
                    ),
                }
            )

            print(
                f"  {stream_name}: "
                f"AUROC="
                f"{metrics.macro_auroc:.4f}, "
                f"BA="
                f"{metrics.balanced_accuracy:.4f}"
            )

    expected_predictions = validation_masks[
        :,
        :,
        None,
        None,
    ]

    missing_expected = expected_predictions & ~np.isfinite(probability_cache)

    if missing_expected.any():
        raise RuntimeError("Expected OOF probabilities are missing.")

    unexpected_predictions = ~validation_masks[
        :,
        :,
        None,
        None,
    ] & np.isfinite(probability_cache)

    if unexpected_predictions.any():
        raise RuntimeError("Predictions were stored outside validation folds.")

    prediction_counts = validation_masks.sum(axis=0)

    if not np.all(prediction_counts == N_REPEATS):
        raise RuntimeError(
            f"Each train patient must receive exactly {N_REPEATS} OOF predictions."
        )

    np.savez_compressed(
        OUTPUT_DIR / "oof_probabilities.npz",
        patient_ids=np.asarray(
            train_patient_ids,
            dtype=np.str_,
        ),
        grouping_ids=np.asarray(
            train_grouping_ids,
            dtype=np.str_,
        ),
        labels=train_labels.astype(np.int64),
        class_names=np.asarray(
            CLASS_NAMES,
            dtype=np.str_,
        ),
        stream_names=np.asarray(
            STREAM_NAMES,
            dtype=np.str_,
        ),
        validation_masks=(validation_masks),
        probabilities=(probability_cache),
    )

    metric_frame = pd.DataFrame(metric_records)

    metric_frame.to_csv(
        OUTPUT_DIR / "oof_stream_metrics.csv",
        index=False,
    )

    assignment_frame = pd.DataFrame(assignment_records)

    assignment_frame.to_csv(
        OUTPUT_DIR / "oof_fold_assignments.csv",
        index=False,
    )

    summary = (
        metric_frame.groupby(
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
                "composite_score",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "oof_stream_summary.csv",
        index=False,
    )

    provenance = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "fold_count": fold_count,
        "train_patients": len(train_labels),
        "streams": STREAM_NAMES,
        "classes": CLASS_NAMES,
        "validation_set_evaluated": False,
        "test_set_evaluated": False,
        "purpose": ("Train-only OOF predictions for fusion-agent search."),
    }

    (OUTPUT_DIR / "oof_provenance.json").write_text(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=== OOF stream summary ===")
    print(summary.to_string(index=False))
    print()
    print(
        "Probability cache shape:",
        probability_cache.shape,
    )
    print(
        "Predictions per patient:",
        sorted(set(prediction_counts.tolist())),
    )
    print("Validation set was not evaluated.")
    print("Test set was not evaluated.")
    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
