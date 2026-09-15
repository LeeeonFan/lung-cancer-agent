import numpy as np

from lung_fusion_agent.baselines.data import BaselineDataset

FOUNDATION_MODEL_NAMES = (
    "uni2",
    "virchow2",
    "prism2",
)


def l2_normalize_rows(
    features: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """L2-normalize each patient embedding independently."""
    if features.ndim != 2:
        raise ValueError("Features must be a two-dimensional array.")

    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinity.")

    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True,
    )

    if (norms < epsilon).any():
        raise ValueError("At least one feature row has a near-zero L2 norm.")

    normalized = features / norms

    return normalized.astype(
        np.float32,
        copy=False,
    )


def build_early_fusion_representations(
    dataset: BaselineDataset,
) -> dict[str, np.ndarray]:
    """Build label-free early-fusion feature matrices."""
    missing_representations = (set(FOUNDATION_MODEL_NAMES) | {"metadata"}) - set(
        dataset.representations
    )

    if missing_representations:
        raise ValueError(f"Missing representations: {sorted(missing_representations)}")

    raw_streams = [
        dataset.representations[model_name] for model_name in FOUNDATION_MODEL_NAMES
    ]

    normalized_streams = [
        l2_normalize_rows(dataset.representations[model_name])
        for model_name in FOUNDATION_MODEL_NAMES
    ]

    metadata = dataset.representations["metadata"]

    early_concat = np.column_stack(
        [
            *raw_streams,
            metadata,
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    early_l2_concat = np.column_stack(
        [
            *normalized_streams,
            metadata,
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    representations = {
        "early_concat": early_concat,
        "early_l2_concat": early_l2_concat,
    }

    for representation_name, features in representations.items():
        if features.ndim != 2:
            raise ValueError(
                f"{representation_name}: features must be two-dimensional."
            )

        if len(features) != len(dataset.patient_ids):
            raise ValueError(f"{representation_name}: patient count mismatch.")

        if not np.isfinite(features).all():
            raise ValueError(
                f"{representation_name}: features contain NaN or infinity."
            )

    return representations
