from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
)

from lung_fusion_agent.baselines.data import (
    BaselineDataset,
)
from lung_fusion_agent.baselines.search import (
    SearchConfig,
    build_classifier,
)

STACKING_STREAM_NAMES = [
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
]


@dataclass(frozen=True)
class StackingFold:
    split_index: int
    repeat_index: int
    fold_index: int
    meta_train_features: np.ndarray
    meta_train_labels: np.ndarray
    meta_validation_features: np.ndarray
    meta_validation_labels: np.ndarray


def _check_class_order(
    model: object,
    *,
    class_count: int,
) -> None:
    classifier = model.named_steps["classifier"]

    expected_classes = np.arange(class_count)

    if not np.array_equal(
        classifier.classes_,
        expected_classes,
    ):
        raise RuntimeError("Unexpected classifier class order.")


def build_nested_stacking_folds(
    *,
    dataset: BaselineDataset,
    train_mask: np.ndarray,
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    selected_configs: dict[
        str,
        SearchConfig,
    ],
    class_count: int,
    outer_n_splits: int,
    inner_n_splits: int,
    seed: int,
) -> list[StackingFold]:
    train_labels = dataset.labels[train_mask]

    stream_features = {
        stream_name: (dataset.representations[stream_name][train_mask])
        for stream_name in (STACKING_STREAM_NAMES)
    }

    stacking_folds = []

    for split_index, (
        outer_fit_indices,
        outer_validation_indices,
    ) in enumerate(outer_folds):
        repeat_index = split_index // outer_n_splits
        fold_index = split_index % outer_n_splits

        outer_fit_labels = train_labels[outer_fit_indices]

        inner_splitter = StratifiedKFold(
            n_splits=inner_n_splits,
            shuffle=True,
            random_state=(seed + split_index),
        )

        inner_folds = list(
            inner_splitter.split(
                np.zeros(len(outer_fit_indices)),
                outer_fit_labels,
            )
        )

        meta_train_parts = []
        meta_validation_parts = []

        print()
        print(f"Building stacking fold {split_index + 1}/{len(outer_folds)}")

        for stream_name in STACKING_STREAM_NAMES:
            print(
                "  Stream:",
                stream_name,
            )

            features = stream_features[stream_name]

            config = selected_configs[stream_name]

            inner_oof_probabilities = np.full(
                (
                    len(outer_fit_indices),
                    class_count,
                ),
                np.nan,
                dtype=np.float32,
            )

            for (
                inner_fit_positions,
                inner_validation_positions,
            ) in inner_folds:
                inner_fit_indices = outer_fit_indices[inner_fit_positions]

                inner_validation_indices = outer_fit_indices[inner_validation_positions]

                inner_model = build_classifier(
                    config,
                    seed=seed,
                )

                inner_model.fit(
                    features[inner_fit_indices],
                    train_labels[inner_fit_indices],
                )

                _check_class_order(
                    inner_model,
                    class_count=(class_count),
                )

                inner_probabilities = inner_model.predict_proba(
                    features[inner_validation_indices]
                ).astype(np.float32)

                inner_oof_probabilities[inner_validation_positions] = (
                    inner_probabilities
                )

            if not np.isfinite(inner_oof_probabilities).all():
                raise RuntimeError(
                    f"{stream_name}: inner OOF predictions are incomplete."
                )

            outer_model = build_classifier(
                config,
                seed=seed,
            )

            outer_model.fit(
                features[outer_fit_indices],
                train_labels[outer_fit_indices],
            )

            _check_class_order(
                outer_model,
                class_count=class_count,
            )

            outer_validation_probabilities = outer_model.predict_proba(
                features[outer_validation_indices]
            ).astype(np.float32)

            meta_train_parts.append(inner_oof_probabilities)

            meta_validation_parts.append(outer_validation_probabilities)

        meta_train_features = np.column_stack(meta_train_parts).astype(np.float32)

        meta_validation_features = np.column_stack(meta_validation_parts).astype(
            np.float32
        )

        expected_feature_count = len(STACKING_STREAM_NAMES) * class_count

        if meta_train_features.shape != (
            len(outer_fit_indices),
            expected_feature_count,
        ):
            raise RuntimeError("Unexpected stacking training feature shape.")

        if meta_validation_features.shape != (
            len(outer_validation_indices),
            expected_feature_count,
        ):
            raise RuntimeError("Unexpected stacking validation feature shape.")

        stacking_folds.append(
            StackingFold(
                split_index=split_index,
                repeat_index=(repeat_index),
                fold_index=fold_index,
                meta_train_features=(meta_train_features),
                meta_train_labels=(train_labels[outer_fit_indices]),
                meta_validation_features=(meta_validation_features),
                meta_validation_labels=(train_labels[outer_validation_indices]),
            )
        )

    return stacking_folds
