from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

REPO_ID = "kmmuleelab/Lung_Pathology_Image_JPG"
METADATA_PATH = Path("data/raw/CLWD.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def wsi_number(wsi_id: str) -> int:
    return int(str(wsi_id).split("-")[-1])


def sorted_wsi_ids(values: list[str]) -> list[str]:
    return sorted(values, key=wsi_number)


def main() -> None:
    metadata = pd.read_csv(
        METADATA_PATH,
        dtype={
            "SampleNumber": "string",
            "WSI_ID": "string",
            "Sex": "string",
            "Benchmark_Label_7class": "string",
        },
    )

    api = HfApi()
    repo_files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    available_wsi_ids = {
        Path(file_path).stem
        for file_path in repo_files
        if Path(file_path).suffix.lower() in IMAGE_EXTENSIONS
    }

    metadata_wsi_ids = set(metadata["WSI_ID"])
    missing_wsi_ids = metadata_wsi_ids - available_wsi_ids
    unexpected_wsi_ids = available_wsi_ids - metadata_wsi_ids

    print("=== Overall availability ===")
    print(f"Metadata WSI IDs: {len(metadata_wsi_ids)}")
    print(f"Available image IDs: {len(available_wsi_ids)}")
    print(f"Metadata images missing from repository: {len(missing_wsi_ids)}")
    print(f"Repository images absent from metadata: {len(unexpected_wsi_ids)}")
    print()

    missing_rows = metadata.loc[
        metadata["WSI_ID"].isin(missing_wsi_ids),
        [
            "SampleNumber",
            "WSI_ID",
            "Age",
            "Sex",
            "Benchmark_Label_7class",
        ],
    ].copy()

    missing_rows["wsi_number"] = missing_rows["WSI_ID"].map(wsi_number)
    missing_rows = missing_rows.sort_values("wsi_number")

    print("=== All metadata rows with missing images ===")
    print(missing_rows.drop(columns="wsi_number").to_string(index=False))
    print()

    patient_columns = ["SampleNumber", "Age", "Sex"]
    patient_summaries: list[dict[str, object]] = []

    for (sample_number, age, sex), group in metadata.groupby(
        patient_columns,
        sort=False,
        dropna=False,
    ):
        all_ids = sorted_wsi_ids(group["WSI_ID"].tolist())
        available_ids = sorted_wsi_ids(
            [wsi_id for wsi_id in all_ids if wsi_id in available_wsi_ids]
        )

        metadata_lowest = all_ids[0]
        lowest_available = available_ids[0] if available_ids else None

        patient_summaries.append(
            {
                "patient_id": f"{sample_number}_age{age}_{sex}",
                "all_wsi_ids": "|".join(all_ids),
                "available_wsi_ids": "|".join(available_ids),
                "metadata_lowest": metadata_lowest,
                "lowest_available": lowest_available,
                "requires_fallback": (
                    lowest_available is not None and metadata_lowest != lowest_available
                ),
                "has_no_available_image": len(available_ids) == 0,
            }
        )

    summary = pd.DataFrame(patient_summaries)

    fallback_patients = summary.loc[summary["requires_fallback"]]
    unavailable_patients = summary.loc[summary["has_no_available_image"]]

    print("=== Patients requiring a fallback slide ===")
    print(f"Count: {len(fallback_patients)}")
    if fallback_patients.empty:
        print("None")
    else:
        print(fallback_patients.to_string(index=False))
    print()

    print("=== Patients with no available image ===")
    print(f"Count: {len(unavailable_patients)}")
    if unavailable_patients.empty:
        print("None")
    else:
        print(unavailable_patients.to_string(index=False))


if __name__ == "__main__":
    main()
