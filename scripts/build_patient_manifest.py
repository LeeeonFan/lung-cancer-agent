from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/raw/CLWD.csv")
OUTPUT_PATH = Path("artifacts/manifests/patient_manifest.csv")
TARGET_COLUMN = "Benchmark_Label_7class"


def extract_wsi_number(wsi_id: str) -> int:
    """Convert WSI-346 to 346."""
    return int(str(wsi_id).split("-")[-1])


def main() -> None:
    df = pd.read_csv(
        INPUT_PATH,
        dtype={
            "SampleNumber": "string",
            "WSI_ID": "string",
            "Sex": "string",
            TARGET_COLUMN: "string",
        },
    )

    # Convert WSI-346 into integer 346 so selection uses numeric order.
    df["wsi_number"] = df["WSI_ID"].map(extract_wsi_number)

    # SampleNumber 8377886 represents two apparent patient records,
    # distinguished by age. Including age and sex recovers 210 records.
    patient_columns = ["SampleNumber", "Age", "Sex"]

    records: list[dict[str, object]] = []

    for (sample_number, age, sex), group in df.groupby(
        patient_columns,
        sort=False,
        dropna=False,
    ):
        group = group.sort_values("wsi_number")
        selected = group.iloc[0]

        labels_seen = sorted(group[TARGET_COLUMN].unique().tolist())
        has_label_conflict = len(labels_seen) > 1

        patient_id = f"{sample_number}_age{age}_{sex}"

        records.append(
            {
                "patient_id": patient_id,
                # This remains the required split grouping key.
                "grouping_id": sample_number,
                "selected_wsi_id": selected["WSI_ID"],
                "selected_wsi_number": selected["wsi_number"],
                "age": int(age),
                "sex": sex,
                "label": selected[TARGET_COLUMN],
                "n_slides_available": len(group),
                "has_label_conflict": has_label_conflict,
                "labels_seen": "|".join(labels_seen),
            }
        )

    manifest = pd.DataFrame(records)
    manifest = manifest.sort_values("patient_id").reset_index(drop=True)

    # Safety checks
    assert len(manifest) == 210, f"Expected 210 patient records, found {len(manifest)}"
    assert manifest["patient_id"].is_unique
    assert manifest["selected_wsi_id"].is_unique
    assert manifest["label"].nunique() == 7
    assert manifest["age"].between(24, 80).all()
    assert set(manifest["sex"]) == {"Female", "Male"}

    # The duplicated SampleNumber must remain one grouping unit.
    duplicated_group = manifest.loc[manifest["grouping_id"] == "8377886"]
    assert len(duplicated_group) == 2

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False)

    print("=== Manifest created ===")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Patient records: {len(manifest)}")
    print(f"Unique grouping IDs: {manifest['grouping_id'].nunique()}")
    print(f"Selected WSIs: {manifest['selected_wsi_id'].nunique()}")
    print()

    print("=== Patient-level class distribution ===")
    print(manifest["label"].value_counts().to_string())
    print()

    print("=== Patient-level sex distribution ===")
    print(manifest["sex"].value_counts().to_string())
    print()

    print("=== Patient-level age summary ===")
    print(manifest["age"].describe().to_string())
    print()

    print("=== Label-conflicting records ===")
    conflicts = manifest.loc[
        manifest["has_label_conflict"],
        [
            "patient_id",
            "selected_wsi_id",
            "label",
            "labels_seen",
        ],
    ]
    print(conflicts.to_string(index=False))
    print()

    print("=== Shared grouping ID 8377886 ===")
    print(duplicated_group.to_string(index=False))


if __name__ == "__main__":
    main()
