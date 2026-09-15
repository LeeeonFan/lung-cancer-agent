import numpy as np

from lung_fusion_agent.baselines.data import (
    load_baseline_dataset,
)
from lung_fusion_agent.fusion.features import (
    FOUNDATION_MODEL_NAMES,
    build_early_fusion_representations,
)


def main() -> None:
    dataset = load_baseline_dataset()

    fusion_representations = build_early_fusion_representations(dataset)

    print("=== Source representations ===")

    for model_name in FOUNDATION_MODEL_NAMES:
        features = dataset.representations[model_name]

        norms = np.linalg.norm(
            features,
            axis=1,
        )

        print(f"{model_name}: shape={features.shape}, mean_l2_norm={norms.mean():.4f}")

    metadata = dataset.representations["metadata"]

    print(
        "metadata:",
        f"shape={metadata.shape}",
    )

    print()
    print("=== Early-fusion representations ===")

    for representation_name, features in fusion_representations.items():
        print(
            f"{representation_name}: "
            f"shape={features.shape}, "
            f"dtype={features.dtype}, "
            f"finite={bool(np.isfinite(features).all())}"
        )

    expected_dimension = (
        sum(
            dataset.representations[model_name].shape[1]
            for model_name in FOUNDATION_MODEL_NAMES
        )
        + metadata.shape[1]
    )

    print()
    print(
        "Expected fused dimension:",
        expected_dimension,
    )

    for representation_name, features in fusion_representations.items():
        if features.shape != (
            len(dataset.patient_ids),
            expected_dimension,
        ):
            raise RuntimeError(f"{representation_name}: unexpected fused shape.")

    print("Fusion input validation passed.")
    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
