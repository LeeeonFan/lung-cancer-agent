import json
from math import sqrt
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
)
from lung_fusion_agent.eval.metrics import (
    calculate_classification_metrics,
)
from lung_fusion_agent.fusion.gated import (
    MetadataGatedLogitFusion,
)
from lung_fusion_agent.fusion.stacking import (
    STACKING_STREAM_NAMES,
    build_nested_stacking_folds,
)

SEED = 42
OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 3
INNER_N_SPLITS = 4

REGULARIZATION_VALUES = [
    0.1,
    1.0,
    10.0,
    100.0,
]

BALANCE_OPTIONS = [
    False,
    True,
]

PRIOR_WEIGHTS = np.asarray(
    [
        0.05,
        0.25,
        0.05,
        0.65,
    ],
    dtype=np.float64,
)

BASELINE_CONFIG_PATH = Path(
    "artifacts/results/baselines/"
    "selected_configs.json"
)

OUTPUT_DIR = Path(
    "artifacts/results/fusion_agent"
)


def load_selected_configs() -> dict[
    str,
    SearchConfig,
]:
    payload = json.loads(
        BASELINE_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    raw_configs = payload[
        "selected_configs"
    ]

    return {
        stream_name: SearchConfig(
            c_value=float(
                raw_configs[
                    stream_name
                ]["c_value"]
            ),
            class_weight=(
                raw_configs[
                    stream_name
                ]["class_weight"]
            ),
        )
        for stream_name in (
            STACKING_STREAM_NAMES
        )
    }


def main() -> None:
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_baseline_dataset()

    train_mask = dataset.mask_for_split(
        "train"
    )

    train_labels = dataset.labels[
        train_mask
    ]

    train_grouping_ids = (
        dataset.grouping_ids[
            train_mask
        ]
    )

    if (
        len(
            np.unique(
                train_grouping_ids
            )
        )
        != len(train_grouping_ids)
    ):
        raise ValueError(
            "Train grouping IDs are not unique."
        )

    outer_splitter = (
        RepeatedStratifiedKFold(
            n_splits=OUTER_N_SPLITS,
            n_repeats=OUTER_N_REPEATS,
            random_state=SEED,
        )
    )

    outer_folds = list(
        outer_splitter.split(
            np.zeros(
                len(train_labels)
            ),
            train_labels,
        )
    )

    stacking_folds = (
        build_nested_stacking_folds(
            dataset=dataset,
            train_mask=train_mask,
            outer_folds=outer_folds,
            selected_configs=(
                load_selected_configs()
            ),
            class_count=len(
                CLASS_NAMES
            ),
            outer_n_splits=(
                OUTER_N_SPLITS
            ),
            inner_n_splits=(
                INNER_N_SPLITS
            ),
            seed=SEED,
        )
    )

    trial_records = []
    fold_records = []

    trial_index = 0

    for regularization in (
        REGULARIZATION_VALUES
    ):
        for class_weight_balanced in (
            BALANCE_OPTIONS
        ):
            print()
            print(
                "=" * 70
            )
            print(
                f"Gated trial "
                f"{trial_index + 1}/8: "
                f"regularization="
                f"{regularization}, "
                f"balanced="
                f"{class_weight_balanced}"
            )

            current_fold_records = []

            for stacking_fold in (
                stacking_folds
            ):
                train_probabilities = (
                    stacking_fold
                    .meta_train_features
                    .reshape(
                        -1,
                        len(
                            STACKING_STREAM_NAMES
                        ),
                        len(
                            CLASS_NAMES
                        ),
                    )
                )

                validation_probabilities = (
                    stacking_fold
                    .meta_validation_features
                    .reshape(
                        -1,
                        len(
                            STACKING_STREAM_NAMES
                        ),
                        len(
                            CLASS_NAMES
                        ),
                    )
                )

                model = (
                    MetadataGatedLogitFusion(
                        prior_weights=(
                            PRIOR_WEIGHTS
                        ),
                        regularization=(
                            regularization
                        ),
                        class_weight_balanced=(
                            class_weight_balanced
                        ),
                    )
                )

                model.fit(
                    metadata=(
                        stacking_fold
                        .gate_train_metadata
                    ),
                    probabilities=(
                        train_probabilities
                    ),
                    labels=(
                        stacking_fold
                        .meta_train_labels
                    ),
                )

                fused_probabilities = (
                    model.predict_proba(
                        metadata=(
                            stacking_fold
                            .gate_validation_metadata
                        ),
                        probabilities=(
                            validation_probabilities
                        ),
                    )
                )

                gates = model.predict_gates(
                    stacking_fold
                    .gate_validation_metadata
                )

                metrics = (
                    calculate_classification_metrics(
                        y_true=(
                            stacking_fold
                            .meta_validation_labels
                        ),
                        probabilities=(
                            fused_probabilities
                        ),
                        class_names=(
                            CLASS_NAMES
                        ),
                    )
                )

                composite_score = (
                    metrics.macro_auroc
                    + metrics.balanced_accuracy
                ) / 2.0

                record = {
                    "trial_index": (
                        trial_index
                    ),
                    "regularization": (
                        regularization
                    ),
                    "class_weight_balanced": (
                        class_weight_balanced
                    ),
                    "split_index": (
                        stacking_fold
                        .split_index
                    ),
                    "repeat_index": (
                        stacking_fold
                        .repeat_index
                    ),
                    "fold_index": (
                        stacking_fold
                        .fold_index
                    ),
                    "macro_auroc": (
                        metrics.macro_auroc
                    ),
                    "balanced_accuracy": (
                        metrics
                        .balanced_accuracy
                    ),
                    "composite_score": (
                        composite_score
                    ),
                    "optimization_success": (
                        model
                        .optimization_success_
                    ),
                    "mean_gate_metadata": (
                        gates[:, 0].mean()
                    ),
                    "mean_gate_uni2": (
                        gates[:, 1].mean()
                    ),
                    "mean_gate_virchow2": (
                        gates[:, 2].mean()
                    ),
                    "mean_gate_prism2": (
                        gates[:, 3].mean()
                    ),
                }

                fold_records.append(
                    record
                )
                current_fold_records.append(
                    record
                )

            current_frame = pd.DataFrame(
                current_fold_records
            )

            composite_mean = float(
                current_frame[
                    "composite_score"
                ].mean()
            )

            composite_std = float(
                current_frame[
                    "composite_score"
                ].std(
                    ddof=1
                )
            )

            composite_sem = (
                composite_std
                / sqrt(
                    len(current_frame)
                )
            )

            trial_records.append(
                {
                    "trial_index": (
                        trial_index
                    ),
                    "regularization": (
                        regularization
                    ),
                    "class_weight_balanced": (
                        class_weight_balanced
                    ),
                    "macro_auroc_mean": (
                        current_frame[
                            "macro_auroc"
                        ].mean()
                    ),
                    "macro_auroc_std": (
                        current_frame[
                            "macro_auroc"
                        ].std(
                            ddof=1
                        )
                    ),
                    "balanced_accuracy_mean": (
                        current_frame[
                            "balanced_accuracy"
                        ].mean()
                    ),
                    "balanced_accuracy_std": (
                        current_frame[
                            "balanced_accuracy"
                        ].std(
                            ddof=1
                        )
                    ),
                    "composite_mean": (
                        composite_mean
                    ),
                    "composite_sem": (
                        composite_sem
                    ),
                    "selection_score": (
                        composite_mean
                        - 0.5
                        * composite_sem
                    ),
                    "optimization_success_rate": (
                        current_frame[
                            "optimization_success"
                        ].mean()
                    ),
                    "mean_gate_metadata": (
                        current_frame[
                            "mean_gate_metadata"
                        ].mean()
                    ),
                    "mean_gate_uni2": (
                        current_frame[
                            "mean_gate_uni2"
                        ].mean()
                    ),
                    "mean_gate_virchow2": (
                        current_frame[
                            "mean_gate_virchow2"
                        ].mean()
                    ),
                    "mean_gate_prism2": (
                        current_frame[
                            "mean_gate_prism2"
                        ].mean()
                    ),
                }
            )

            trial_index += 1

    trial_frame = (
        pd.DataFrame(
            trial_records
        )
        .sort_values(
            "selection_score",
            ascending=False,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    fold_frame = pd.DataFrame(
        fold_records
    )

    trial_frame.to_csv(
        OUTPUT_DIR
        / "gated_fusion_trials.csv",
        index=False,
    )

    fold_frame.to_csv(
        OUTPUT_DIR
        / "gated_fusion_fold_metrics.csv",
        index=False,
    )

    best_trial = (
        trial_frame.iloc[0].to_dict()
    )

    summary = {
        "strategy": (
            "metadata_conditioned_logit_gate"
        ),
        "prior_weights": (
            PRIOR_WEIGHTS.tolist()
        ),
        "gate_parameter_count": 12,
        "outer_splits": (
            OUTER_N_SPLITS
        ),
        "outer_repeats": (
            OUTER_N_REPEATS
        ),
        "inner_splits": (
            INNER_N_SPLITS
        ),
        "trial_budget": len(
            REGULARIZATION_VALUES
        )
        * len(
            BALANCE_OPTIONS
        ),
        "best_trial": (
            best_trial
        ),
        "validation_set_evaluated": False,
        "test_set_evaluated": False,
    }

    (
        OUTPUT_DIR
        / "gated_fusion_best.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=== Gated fusion results ==="
    )

    print(
        trial_frame.to_string(
            index=False
        )
    )

    print()
    print(
        "=== Best gated trial ==="
    )

    print(
        json.dumps(
            best_trial,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "Validation set was not evaluated."
    )
    print(
        "Test set was not evaluated."
    )


if __name__ == "__main__":
    main()