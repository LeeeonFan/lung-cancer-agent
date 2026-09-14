from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

REPO_ID = "kmmuleelab/Lung_Pathology_Image_JPG"
MANIFEST_PATH = Path("artifacts/manifests/patient_manifest.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".7z"}


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    selected_wsi_ids = set(manifest["selected_wsi_id"].astype(str))

    api = HfApi()

    print("Querying Hugging Face repository file list...")

    repo_files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    print()
    print("=== Repository summary ===")
    print(f"Total repository files: {len(repo_files)}")

    suffix_counts = Counter(
        Path(file_path).suffix.lower() or "[no extension]"
        for file_path in repo_files
    )

    print()
    print("=== File types ===")
    for suffix, count in suffix_counts.most_common():
        print(f"{suffix}: {count}")

    top_level_counts = Counter(
        Path(file_path).parts[0]
        if len(Path(file_path).parts) > 1
        else "[repository root]"
        for file_path in repo_files
    )

    print()
    print("=== Top-level locations ===")
    for location, count in top_level_counts.most_common():
        print(f"{location}: {count}")

    image_files = [
        file_path
        for file_path in repo_files
        if Path(file_path).suffix.lower() in IMAGE_EXTENSIONS
    ]

    archive_files = [
        file_path
        for file_path in repo_files
        if Path(file_path).suffix.lower() in ARCHIVE_EXTENSIONS
    ]

    print()
    print("=== Sample image paths ===")
    for file_path in image_files[:20]:
        print(file_path)

    print()
    print("=== Archive files ===")
    if archive_files:
        for file_path in archive_files:
            print(file_path)
    else:
        print("No archive files found.")

    # Match WSI_ID against the filename without its extension.
    paths_by_stem = {
        Path(file_path).stem: file_path
        for file_path in image_files
    }

    matched_ids = selected_wsi_ids.intersection(paths_by_stem)
    missing_ids = sorted(
        selected_wsi_ids - set(paths_by_stem),
        key=lambda value: int(value.split("-")[-1]),
    )

    print()
    print("=== Selected-image matching ===")
    print(f"Selected WSI IDs in manifest: {len(selected_wsi_ids)}")
    print(f"Matched image files: {len(matched_ids)}")
    print(f"Missing selected image files: {len(missing_ids)}")

    if matched_ids:
        print()
        print("Example matches:")
        for wsi_id in sorted(
            matched_ids,
            key=lambda value: int(value.split("-")[-1]),
        )[:10]:
            print(f"{wsi_id} -> {paths_by_stem[wsi_id]}")

    if missing_ids:
        print()
        print("First missing IDs:")
        for wsi_id in missing_ids[:20]:
            print(wsi_id)


if __name__ == "__main__":
    main()