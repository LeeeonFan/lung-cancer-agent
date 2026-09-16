from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from lung_fusion_agent.baselines.data import BaselineDataset
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
    """Data needed to train and evaluate one outer stacking fold."""

    split_index: int
    repeat_index: int
    fold_index: int

    # OOF probability features used to train
    # the stacking model or gating model.
    meta_train_features: np.ndarray
    meta_train_labels: np.ndarray

    # Base-model probabilities for the outer
    # validation patients.
    meta_validation_features: np.ndarray
    meta_validation_labels: np.ndarray

    # Raw age and sex values used by the
    # metadata-conditioned gating model.
    gate_train_metadata: np.ndarray
    gate_validation_metadata: np.ndarray


def _check_class_order(
    model: object,
    *,
    class_count: int,
) -> None:
    """Confirm that predict_proba columns use the expected class order."""
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
    """Create leakage-safe nested stacking features.

    For each outer fold:

    1. Inner CV produces OOF base-model probabilities
       for the outer-training patients.
    2. Each base model is refitted on the complete
       outer-training subset.
    3. Those refitted models predict the outer-validation
       patients.
    4. Raw age and sex are retained for gated fusion.

    Validation and test splits outside the supplied train_mask
    are never accessed.
    """
    if train_mask.dtype != bool:
        raise ValueError("train_mask must be a boolean array.")

    if train_mask.shape != dataset.labels.shape:
        raise ValueError("train_mask shape does not match dataset labels.")

    train_labels = dataset.labels[train_mask]

    train_metadata = dataset.representations["metadata"][train_mask]

    if train_metadata.shape != (
        len(train_labels),
        2,
    ):
        raise ValueError(
            "Expected train metadata with shape "
            f"({len(train_labels)}, 2), "
            f"found {train_metadata.shape}."
        )

    if not np.isfinite(train_metadata).all():
        raise ValueError("Train metadata contains NaN or infinity.")

    missing_streams = set(STACKING_STREAM_NAMES) - set(dataset.representations)

    if missing_streams:
        raise ValueError(
            f"Dataset is missing stacking streams: {sorted(missing_streams)}"
        )

    missing_configs = set(STACKING_STREAM_NAMES) - set(selected_configs)

    if missing_configs:
        raise ValueError(f"Missing selected configurations: {sorted(missing_configs)}")

    stream_features = {
        stream_name: (dataset.representations[stream_name][train_mask])
        for stream_name in (STACKING_STREAM_NAMES)
    }

    for stream_name, features in stream_features.items():
        if features.ndim != 2:
            raise ValueError(f"{stream_name}: features must be two-dimensional.")

        if len(features) != len(train_labels):
            raise ValueError(f"{stream_name}: train patient count mismatch.")

        if not np.isfinite(features).all():
            raise ValueError(f"{stream_name}: features contain NaN or infinity.")

    stacking_folds = []

    for split_index, (
        outer_fit_indices,
        outer_validation_indices,
    ) in enumerate(outer_folds):
        repeat_index = split_index // outer_n_splits

        fold_index = split_index % outer_n_splits

        if np.intersect1d(
            outer_fit_indices,
            outer_validation_indices,
        ).size:
            raise ValueError("Outer training and validation indices overlap.")

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

        if len(inner_folds) != (inner_n_splits):
            raise RuntimeError("Unexpected number of inner folds.")

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

            inner_prediction_counts = np.zeros(
                len(outer_fit_indices),
                dtype=np.int64,
            )

            for (
                inner_fit_positions,
                inner_validation_positions,
            ) in inner_folds:
                if np.intersect1d(
                    inner_fit_positions,
                    inner_validation_positions,
                ).size:
                    raise ValueError("Inner training and validation positions overlap.")

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

                expected_inner_shape = (
                    len(inner_validation_positions),
                    class_count,
                )

                if inner_probabilities.shape != expected_inner_shape:
                    raise RuntimeError(
                        f"{stream_name}: unexpected inner probability shape."
                    )

                inner_oof_probabilities[inner_validation_positions] = (
                    inner_probabilities
                )

                inner_prediction_counts[inner_validation_positions] += 1

            if not np.all(inner_prediction_counts == 1):
                raise RuntimeError(
                    f"{stream_name}: each outer-training "
                    "patient must receive exactly one "
                    "inner OOF prediction."
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
                class_count=(class_count),
            )

            outer_validation_probabilities = outer_model.predict_proba(
                features[outer_validation_indices]
            ).astype(np.float32)

            expected_outer_shape = (
                len(outer_validation_indices),
                class_count,
            )

            if outer_validation_probabilities.shape != expected_outer_shape:
                raise RuntimeError(
                    f"{stream_name}: unexpected outer-validation probability shape."
                )

            if not np.isfinite(outer_validation_probabilities).all():
                raise RuntimeError(
                    f"{stream_name}: outer-validation "
                    "probabilities contain NaN or infinity."
                )

            meta_train_parts.append(inner_oof_probabilities)

            meta_validation_parts.append(outer_validation_probabilities)

        meta_train_features = np.column_stack(meta_train_parts).astype(
            np.float32,
            copy=False,
        )

        meta_validation_features = np.column_stack(meta_validation_parts).astype(
            np.float32,
            copy=False,
        )

        expected_feature_count = len(STACKING_STREAM_NAMES) * class_count

        expected_train_shape = (
            len(outer_fit_indices),
            expected_feature_count,
        )

        expected_validation_shape = (
            len(outer_validation_indices),
            expected_feature_count,
        )

        if meta_train_features.shape != expected_train_shape:
            raise RuntimeError(
                "Unexpected stacking training "
                "feature shape: "
                f"{meta_train_features.shape}; "
                f"expected {expected_train_shape}."
            )

        if meta_validation_features.shape != expected_validation_shape:
            raise RuntimeError(
                "Unexpected stacking validation "
                "feature shape: "
                f"{meta_validation_features.shape}; "
                f"expected {expected_validation_shape}."
            )

        gate_train_metadata = train_metadata[outer_fit_indices].astype(
            np.float32,
            copy=False,
        )

        gate_validation_metadata = train_metadata[outer_validation_indices].astype(
            np.float32,
            copy=False,
        )

        if gate_train_metadata.shape != (
            len(outer_fit_indices),
            2,
        ):
            raise RuntimeError("Unexpected gate training metadata shape.")

        if gate_validation_metadata.shape != (
            len(outer_validation_indices),
            2,
        ):
            raise RuntimeError("Unexpected gate validation metadata shape.")

        stacking_folds.append(
            StackingFold(
                split_index=split_index,
                repeat_index=(repeat_index),
                fold_index=(fold_index),
                meta_train_features=(meta_train_features),
                meta_train_labels=(
                    train_labels[outer_fit_indices].astype(
                        np.int64,
                        copy=False,
                    )
                ),
                meta_validation_features=(meta_validation_features),
                meta_validation_labels=(
                    train_labels[outer_validation_indices].astype(
                        np.int64,
                        copy=False,
                    )
                ),
                gate_train_metadata=(gate_train_metadata),
                gate_validation_metadata=(gate_validation_metadata),
            )
        )

    if len(stacking_folds) != len(outer_folds):
        raise RuntimeError("Unexpected number of completed stacking folds.")

    return stacking_folds
