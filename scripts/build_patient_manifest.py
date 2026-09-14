from pathlib import Path

import pandas as pd

METADATA_PATH = Path("data/raw/CLWD.csv")
INVENTORY_PATH = Path("artifacts/manifests/dataset_image_inventory.csv")
MANIFEST_PATH = Path("artifacts/manifests/patient_manifest.csv")
EXCLUSIONS_PATH = Path("artifacts/manifests/excluded_patients.csv")

TARGET_COLUMN = "Benchmark_Label_7class"


def extract_wsi_number(wsi_id: str) -> int:
    """Convert an ID such as WSI-346 to the integer 346."""
    return int(str(wsi_id).split("-")[-1])


def make_patient_id(
    sample_number: str,
    age: int,
    sex: str,
) -> str:
    """Create a unique ID for each apparent patient record."""
    return f"{sample_number}_age{age}_{sex}"


def main() -> None:
    metadata = pd.read_csv(
        METADATA_PATH,
        dtype={
            "SampleNumber": "string",
            "WSI_ID": "string",
            "Sex": "string",
            TARGET_COLUMN: "string",
        },
    )

    inventory = pd.read_csv(
        INVENTORY_PATH,
        dtype={
            "wsi_id": "string",
            "repo_path": "string",
        },
    )

    available_paths = dict(
        zip(
            inventory["wsi_id"],
            inventory["repo_path"],
            strict=True,
        )
    )
    available_wsi_ids = set(available_paths)

    metadata["wsi_number"] = metadata["WSI_ID"].map(extract_wsi_number)

    patient_columns = ["SampleNumber", "Age", "Sex"]

    included_records: list[dict[str, object]] = []
    excluded_records: list[dict[str, object]] = []

    for (sample_number, age, sex), group in metadata.groupby(
        patient_columns,
        sort=False,
        dropna=False,
    ):
        group = group.sort_values("wsi_number")

        patient_id = make_patient_id(
            str(sample_number),
            int(age),
            str(sex),
        )

        all_wsi_ids = group["WSI_ID"].tolist()
        available_group = group.loc[group["WSI_ID"].isin(available_wsi_ids)].copy()

        labels_seen = sorted(group[TARGET_COLUMN].unique().tolist())

        if available_group.empty:
            excluded_records.append(
                {
                    "patient_id": patient_id,
                    "grouping_id": sample_number,
                    "age": int(age),
                    "sex": sex,
                    "all_wsi_ids": "|".join(all_wsi_ids),
                    "labels_seen": "|".join(labels_seen),
                    "exclusion_reason": ("no_image_available_in_pinned_revision"),
                }
            )
            continue

        selected = available_group.iloc[0]
        metadata_lowest_wsi = group.iloc[0]["WSI_ID"]
        selected_wsi_id = selected["WSI_ID"]

        included_records.append(
            {
                "patient_id": patient_id,
                "grouping_id": sample_number,
                "selected_wsi_id": selected_wsi_id,
                "repo_path": available_paths[selected_wsi_id],
                "selected_wsi_number": int(selected["wsi_number"]),
                "age": int(age),
                "sex": sex,
                "label": selected[TARGET_COLUMN],
                "n_slides_in_metadata": len(group),
                "n_slides_available": len(available_group),
                "metadata_lowest_wsi": metadata_lowest_wsi,
                "used_fallback_slide": (selected_wsi_id != metadata_lowest_wsi),
                "has_label_conflict": len(labels_seen) > 1,
                "labels_seen": "|".join(labels_seen),
            }
        )

    manifest = pd.DataFrame(included_records)
    exclusions = pd.DataFrame(excluded_records)

    manifest = manifest.sort_values("patient_id").reset_index(drop=True)
    exclusions = exclusions.sort_values("patient_id").reset_index(drop=True)

    # Safety checks for the pinned repository revision.
    assert len(manifest) == 204, f"Expected 204 included records, found {len(manifest)}"
    assert len(exclusions) == 6, f"Expected 6 exclusions, found {len(exclusions)}"
    assert len(manifest) + len(exclusions) == 210
    assert manifest["patient_id"].is_unique
    assert manifest["selected_wsi_id"].is_unique
    assert manifest["label"].nunique() == 7
    assert set(manifest["selected_wsi_id"]).issubset(available_wsi_ids)

    # The two records sharing SampleNumber 8377886 remain included
    # but must later be assigned to the same data split.
    duplicated_group = manifest.loc[manifest["grouping_id"] == "8377886"]
    assert len(duplicated_group) == 2

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    exclusions.to_csv(EXCLUSIONS_PATH, index=False)

    print("=== Availability-aware manifest created ===")
    print(f"Included patients: {len(manifest)}")
    print(f"Excluded patients: {len(exclusions)}")
    print(f"Unique grouping IDs: {manifest['grouping_id'].nunique()}")
    print(f"Selected available images: {manifest['selected_wsi_id'].nunique()}")
    print()

    print("=== Patient-level class distribution ===")
    print(manifest["label"].value_counts().to_string())
    print()

    print("=== Fallback slides ===")
    fallback = manifest.loc[
        manifest["used_fallback_slide"],
        [
            "patient_id",
            "metadata_lowest_wsi",
            "selected_wsi_id",
            "label",
        ],
    ]
    print(f"Count: {len(fallback)}")
    print(fallback.to_string(index=False))
    print()

    print("=== Excluded patients ===")
    print(
        exclusions[
            [
                "patient_id",
                "all_wsi_ids",
                "labels_seen",
                "exclusion_reason",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
