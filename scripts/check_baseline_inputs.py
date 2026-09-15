import numpy as np

from lung_fusion_agent.baselines.data import (
    CLASS_NAMES,
    load_baseline_dataset,
)


def main() -> None:
    dataset = load_baseline_dataset()

    print("=== Baseline input summary ===")
    print("Patients:", len(dataset.patient_ids))
    print("Classes:", CLASS_NAMES)

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        mask = dataset.mask_for_split(split_name)

        print()
        print(f"{split_name}: {int(mask.sum())}")

        labels, counts = np.unique(
            dataset.label_names[mask],
            return_counts=True,
        )

        for label, count in zip(
            labels,
            counts,
            strict=True,
        ):
            print(f"  {label}: {count}")

    print()
    print("=== Representations ===")

    for name, matrix in dataset.representations.items():
        print(
            f"{name}: "
            f"shape={matrix.shape}, "
            f"dtype={matrix.dtype}, "
            f"finite="
            f"{bool(np.isfinite(matrix).all())}"
        )


if __name__ == "__main__":
    main()
