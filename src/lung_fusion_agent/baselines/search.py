from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lung_fusion_agent.eval.metrics import (
    calculate_classification_metrics,
)


@dataclass(frozen=True)
class SearchConfig:
    c_value: float
    class_weight: str | None


C_VALUES = [
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
]

CLASS_WEIGHTS = [
    None,
    "balanced",
]


def build_search_configs() -> list[SearchConfig]:
    return [
        SearchConfig(
            c_value=c_value,
            class_weight=class_weight,
        )
        for c_value in C_VALUES
        for class_weight in CLASS_WEIGHTS
    ]


def build_classifier(
    config: SearchConfig,
    *,
    seed: int,
) -> Pipeline:
    classifier = LogisticRegression(
        C=config.c_value,
        class_weight=config.class_weight,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-5,
        random_state=seed,
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def run_repeated_cv_search(
    *,
    representation_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    SearchConfig,
]:
    if not np.isfinite(features).all():
        raise ValueError(f"{representation_name}: features contain NaN or infinity.")

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    folds = list(
        splitter.split(
            np.zeros(len(labels)),
            labels,
        )
    )

    expected_fold_count = n_splits * n_repeats

    if len(folds) != expected_fold_count:
        raise RuntimeError("Unexpected number of CV folds.")

    fold_records = []
    configs = build_search_configs()

    for trial_index, config in enumerate(configs):
        print(
            f"{representation_name}: "
            f"trial {trial_index + 1}/"
            f"{len(configs)}, "
            f"C={config.c_value}, "
            f"class_weight="
            f"{config.class_weight}"
        )

        for split_index, (
            train_indices,
            validation_indices,
        ) in enumerate(folds):
            repeat_index = split_index // n_splits
            fold_index = split_index % n_splits

            model = build_classifier(
                config,
                seed=seed,
            )

            model.fit(
                features[train_indices],
                labels[train_indices],
            )

            classifier = model.named_steps["classifier"]

            expected_classes = np.arange(len(class_names))

            if not np.array_equal(
                classifier.classes_,
                expected_classes,
            ):
                raise RuntimeError(
                    "Classifier class order does not match expected order."
                )

            probabilities = model.predict_proba(features[validation_indices])

            metrics = calculate_classification_metrics(
                y_true=labels[validation_indices],
                probabilities=probabilities,
                class_names=class_names,
            )

            composite_score = (metrics.macro_auroc + metrics.balanced_accuracy) / 2.0

            fold_records.append(
                {
                    "representation": (representation_name),
                    "trial_index": trial_index,
                    "c_value": config.c_value,
                    "class_weight": (
                        config.class_weight
                        if config.class_weight is not None
                        else "none"
                    ),
                    "repeat_index": repeat_index,
                    "fold_index": fold_index,
                    "macro_auroc": (metrics.macro_auroc),
                    "balanced_accuracy": (metrics.balanced_accuracy),
                    "composite_score": (composite_score),
                }
            )

    fold_frame = pd.DataFrame(fold_records)

    summary_records = []

    for trial_index, trial_frame in fold_frame.groupby("trial_index"):
        first_row = trial_frame.iloc[0]

        composite_mean = trial_frame["composite_score"].mean()
        composite_std = trial_frame["composite_score"].std(ddof=1)
        composite_sem = composite_std / sqrt(len(trial_frame))

        # Variance-aware selection:
        # reward mean performance while
        # penalizing unstable configurations.
        selection_score = composite_mean - 0.5 * composite_sem

        summary_records.append(
            {
                "representation": (representation_name),
                "trial_index": int(trial_index),
                "c_value": float(first_row["c_value"]),
                "class_weight": first_row["class_weight"],
                "fold_count": len(trial_frame),
                "macro_auroc_mean": (trial_frame["macro_auroc"].mean()),
                "macro_auroc_std": (trial_frame["macro_auroc"].std(ddof=1)),
                "balanced_accuracy_mean": (trial_frame["balanced_accuracy"].mean()),
                "balanced_accuracy_std": (trial_frame["balanced_accuracy"].std(ddof=1)),
                "composite_mean": (composite_mean),
                "composite_std": (composite_std),
                "composite_sem": (composite_sem),
                "selection_score": (selection_score),
            }
        )

    summary_frame = (
        pd.DataFrame(summary_records)
        .sort_values(
            [
                "selection_score",
                "macro_auroc_mean",
                "balanced_accuracy_mean",
            ],
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    best_row = summary_frame.iloc[0]

    best_class_weight = (
        None if best_row["class_weight"] == "none" else str(best_row["class_weight"])
    )

    best_config = SearchConfig(
        c_value=float(best_row["c_value"]),
        class_weight=best_class_weight,
    )

    return (
        fold_frame,
        summary_frame,
        best_config,
    )


def config_to_dict(
    config: SearchConfig,
) -> dict[str, object]:
    return asdict(config)
