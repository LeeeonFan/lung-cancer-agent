import json
from pathlib import Path

import numpy as np
import pandas as pd

from lung_fusion_agent.baselines.data import (
    CLASS_NAMES,
    load_baseline_dataset,
)
from lung_fusion_agent.baselines.search import (
    build_classifier,
    config_to_dict,
    run_repeated_cv_search,
)
from lung_fusion_agent.eval.metrics import (
    calculate_classification_metrics,
)

SEED = 42
N_SPLITS = 5
N_REPEATS = 3

REPRESENTATIONS = [
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
]

OUTPUT_DIR = Path("artifacts/results/baselines")


def main() -> None:
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_baseline_dataset()

    train_mask = dataset.mask_for_split("train")
    validation_mask = dataset.mask_for_split("validation")

    train_labels = dataset.labels[train_mask]
    validation_labels = dataset.labels[validation_mask]

    all_fold_frames = []
    all_summary_frames = []
    validation_records = []
    selected_configs = {}

    for representation_name in REPRESENTATIONS:
        print()
        print("=" * 70)
        print(
            "Representation:",
            representation_name,
        )

        features = dataset.representations[representation_name]

        train_features = features[train_mask]
        validation_features = features[validation_mask]

        (
            fold_frame,
            summary_frame,
            best_config,
        ) = run_repeated_cv_search(
            representation_name=(representation_name),
            features=train_features,
            labels=train_labels,
            class_names=CLASS_NAMES,
            seed=SEED,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
        )

        all_fold_frames.append(fold_frame)
        all_summary_frames.append(summary_frame)

        selected_configs[representation_name] = config_to_dict(best_config)

        final_model = build_classifier(
            best_config,
            seed=SEED,
        )

        # Fit only on train. Validation is
        # evaluated once after CV selection.
        final_model.fit(
            train_features,
            train_labels,
        )

        classifier = final_model.named_steps["classifier"]

        expected_classes = np.arange(len(CLASS_NAMES))

        if not np.array_equal(
            classifier.classes_,
            expected_classes,
        ):
            raise RuntimeError("Unexpected classifier class order.")

        validation_probabilities = final_model.predict_proba(validation_features)

        validation_metrics = calculate_classification_metrics(
            y_true=validation_labels,
            probabilities=(validation_probabilities),
            class_names=CLASS_NAMES,
        )

        validation_record = {
            "representation": (representation_name),
            "c_value": best_config.c_value,
            "class_weight": (
                best_config.class_weight
                if best_config.class_weight is not None
                else "none"
            ),
            "macro_auroc": (validation_metrics.macro_auroc),
            "balanced_accuracy": (validation_metrics.balanced_accuracy),
            "per_class_auroc_json": (
                json.dumps(
                    validation_metrics.per_class_auroc,
                    sort_keys=True,
                )
            ),
        }

        validation_records.append(validation_record)

        print()
        print("Selected configuration:")
        print(
            json.dumps(
                selected_configs[representation_name],
                indent=2,
                sort_keys=True,
            )
        )
        print(
            "Validation macro AUROC:",
            round(
                validation_metrics.macro_auroc,
                4,
            ),
        )
        print(
            "Validation balanced accuracy:",
            round(
                validation_metrics.balanced_accuracy,
                4,
            ),
        )

    all_folds = pd.concat(
        all_fold_frames,
        ignore_index=True,
    )
    all_summaries = pd.concat(
        all_summary_frames,
        ignore_index=True,
    )
    validation_frame = pd.DataFrame(validation_records)

    all_folds.to_csv(
        OUTPUT_DIR / "cv_folds.csv",
        index=False,
    )
    all_summaries.to_csv(
        OUTPUT_DIR / "cv_trials.csv",
        index=False,
    )
    validation_frame.to_csv(
        OUTPUT_DIR / "validation_results.csv",
        index=False,
    )

    selection_metadata = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "trial_count_per_representation": 12,
        "selection_metric": ("mean((macro_auroc + balanced_accuracy) / 2) - 0.5 * SEM"),
        "test_set_evaluated": False,
        "selected_configs": (selected_configs),
    }

    (OUTPUT_DIR / "selected_configs.json").write_text(
        json.dumps(
            selection_metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=== Validation results ===")
    print(
        validation_frame[
            [
                "representation",
                "macro_auroc",
                "balanced_accuracy",
            ]
        ].to_string(index=False)
    )
    print()
    print("Test set was not evaluated.")
    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
