from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("artifacts/manifests/patient_manifest.csv")
OUTPUT_PATH = Path("artifacts/splits/patient_splits.csv")
SEED = 42


def main() -> None:
    manifest = pd.read_csv(
        INPUT_PATH,
        dtype={
            "patient_id": "string",
            "grouping_id": "string",
            "selected_wsi_id": "string",
            "sex": "string",
            "label": "string",
        },
    )

    # Build one row per grouping ID.
    # This ensures SampleNumber 8377886 cannot cross splits.
    group_table = manifest.groupby("grouping_id", as_index=False).agg(
        stratify_label=("label", "first"),
        n_patient_records=("patient_id", "size"),
    )

    # Confirm every grouping ID has only one selected target label.
    labels_per_group = manifest.groupby("grouping_id")["label"].nunique()
    if (labels_per_group > 1).any():
        bad_groups = labels_per_group[labels_per_group > 1]
        raise ValueError(
            f"A grouping ID contains multiple selected labels:\n{bad_groups}"
        )

    # First reserve approximately 20% of grouping units for test.
    development_groups, test_groups = train_test_split(
        group_table,
        test_size=0.20,
        random_state=SEED,
        stratify=group_table["stratify_label"],
    )

    # Validation should be 10% of the full dataset.
    # After removing 20% test, validation is 0.10 / 0.80 = 0.125
    # of the remaining development groups.
    train_groups, validation_groups = train_test_split(
        development_groups,
        test_size=0.125,
        random_state=SEED,
        stratify=development_groups["stratify_label"],
    )

    group_to_split: dict[str, str] = {}

    for grouping_id in train_groups["grouping_id"]:
        group_to_split[str(grouping_id)] = "train"

    for grouping_id in validation_groups["grouping_id"]:
        group_to_split[str(grouping_id)] = "validation"

    for grouping_id in test_groups["grouping_id"]:
        group_to_split[str(grouping_id)] = "test"

    result = manifest.copy()
    result["split"] = result["grouping_id"].map(group_to_split)

    # Safety checks
    assert result["split"].notna().all()
    assert len(result) == 210
    assert result["patient_id"].is_unique
    assert result["selected_wsi_id"].is_unique

    # Every grouping ID must occur in exactly one split.
    splits_per_group = result.groupby("grouping_id")["split"].nunique()
    assert splits_per_group.max() == 1

    # All seven classes should appear in every split.
    class_counts = pd.crosstab(result["label"], result["split"])
    assert set(class_counts.index) == set(result["label"].unique())
    assert (class_counts > 0).all().all()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("=== Fixed patient-level split created ===")
    print(f"Seed: {SEED}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    print("=== Patient records per split ===")
    print(result["split"].value_counts().to_string())
    print()

    print("=== Grouping IDs per split ===")
    print(result.groupby("split")["grouping_id"].nunique().sort_index().to_string())
    print()

    print("=== Class distribution per split ===")
    print(class_counts.to_string())
    print()

    duplicate_id_rows = result.loc[
        result["grouping_id"] == "8377886",
        ["patient_id", "grouping_id", "selected_wsi_id", "split"],
    ]

    print("=== Shared grouping ID 8377886 ===")
    print(duplicate_id_rows.to_string(index=False))


if __name__ == "__main__":
    main()
