import json
from pathlib import Path

import numpy as np
import pandas as pd

from lung_fusion_agent.baselines.data import (
    CLASS_NAMES,
    load_baseline_dataset,
)
from lung_fusion_agent.baselines.search import (
    config_to_dict,
    run_repeated_cv_search,
)
from lung_fusion_agent.fusion.features import (
    build_early_fusion_representations,
)

SEED = 42
N_SPLITS = 5
N_REPEATS = 3

OUTPUT_DIR = Path("artifacts/results/fusion_probe")

REPRESENTATION_NAMES = [
    "early_concat",
    "early_l2_concat",
]


def main() -> None:
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_baseline_dataset()

    fusion_representations = build_early_fusion_representations(dataset)

    train_mask = dataset.mask_for_split("train")
    train_labels = dataset.labels[train_mask]

    fold_frames = []
    summary_frames = []
    selected_configs = {}

    for representation_name in REPRESENTATION_NAMES:
        print()
        print("=" * 70)
        print(
            "Fusion representation:",
            representation_name,
        )

        all_features = fusion_representations[representation_name]

        train_features = all_features[train_mask]

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

        fold_frames.append(fold_frame)
        summary_frames.append(summary_frame)

        selected_configs[representation_name] = config_to_dict(best_config)

        best_result = summary_frame.iloc[0]

        print()
        print("Best train-CV result:")
        print(
            "  C:",
            best_config.c_value,
        )
        print(
            "  class_weight:",
            best_config.class_weight,
        )
        print(
            "  macro AUROC:",
            round(
                float(best_result["macro_auroc_mean"]),
                6,
            ),
        )
        print(
            "  balanced accuracy:",
            round(
                float(best_result["balanced_accuracy_mean"]),
                6,
            ),
        )
        print(
            "  selection score:",
            round(
                float(best_result["selection_score"]),
                6,
            ),
        )

    all_folds = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    all_trials = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    all_folds.to_csv(
        OUTPUT_DIR / "cv_folds.csv",
        index=False,
    )

    all_trials.to_csv(
        OUTPUT_DIR / "cv_trials.csv",
        index=False,
    )

    metadata = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "representations": (REPRESENTATION_NAMES),
        "selected_configs": (selected_configs),
        "selection_metric": ("mean((macro_auroc + balanced_accuracy) / 2) - 0.5 * SEM"),
        "validation_set_evaluated": False,
        "test_set_evaluated": False,
    }

    (OUTPUT_DIR / "probe_metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    best_results = (
        all_trials.sort_values(
            [
                "representation",
                "selection_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "representation",
            sort=False,
        )
        .head(1)
    )

    print()
    print("=== Best early-fusion train-CV results ===")

    print(
        best_results[
            [
                "representation",
                "c_value",
                "class_weight",
                "macro_auroc_mean",
                "macro_auroc_std",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
                "selection_score",
            ]
        ].to_string(index=False)
    )

    print()
    print("Validation set was not evaluated.")
    print("Test set was not evaluated.")
    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
