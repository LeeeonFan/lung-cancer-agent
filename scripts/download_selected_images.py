import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

MANIFEST_PATH = Path("artifacts/manifests/patient_manifest.csv")
PROVENANCE_PATH = Path("artifacts/manifests/dataset_provenance.json")
OUTPUT_DIRECTORY = Path("data/raw/images")


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024

    raise RuntimeError("Unable to format file size")


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate or download selected WSI images "
            "from the pinned Hugging Face revision."
        )
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download files after estimating their total size.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = get_arguments()

    manifest = pd.read_csv(MANIFEST_PATH)
    selected_paths = manifest["repo_path"].tolist()

    with PROVENANCE_PATH.open(encoding="utf-8") as file:
        provenance = json.load(file)

    repo_id = provenance["repo_id"]
    revision = provenance["revision"]

    api = HfApi()

    print("Querying selected image sizes...")

    path_info = api.get_paths_info(
        repo_id=repo_id,
        paths=selected_paths,
        repo_type="dataset",
        revision=revision,
    )

    size_by_path = {item.path: item.size for item in path_info if hasattr(item, "size")}

    missing_info = sorted(set(selected_paths) - set(size_by_path))

    if missing_info:
        raise RuntimeError(f"Could not retrieve file information for: {missing_info}")

    total_size = sum(size_by_path.values())
    largest_paths = sorted(
        size_by_path.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    disk_usage = shutil.disk_usage(OUTPUT_DIRECTORY.parent)
    free_space = disk_usage.free

    print()
    print("=== Download estimate ===")
    print(f"Repository: {repo_id}")
    print(f"Revision: {revision}")
    print(f"Selected images: {len(selected_paths)}")
    print(f"Total download size: {human_size(total_size)}")
    print(f"Available disk space: {human_size(free_space)}")
    print()

    print("=== Five largest selected images ===")
    for path, size in largest_paths[:5]:
        print(f"{path}: {human_size(size)}")

    if not arguments.download:
        print()
        print("Estimate only: no images were downloaded.")
        print("Run again with --download after checking the required disk space.")
        return

    # Keep additional working space available for downloads,
    # temporary files, downsampling and tile processing.
    required_space = int(total_size * 1.25)

    if free_space < required_space:
        raise RuntimeError(
            "Insufficient free disk space. "
            f"Recommended minimum: {human_size(required_space)}"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print()
    print("=== Downloading selected images ===")

    for index, repo_path in enumerate(selected_paths, start=1):
        print(f"[{index}/{len(selected_paths)}] {repo_path}")

        hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="dataset",
            revision=revision,
            local_dir=OUTPUT_DIRECTORY,
        )

    downloaded_files = [OUTPUT_DIRECTORY / repo_path for repo_path in selected_paths]
    missing_downloads = [path for path in downloaded_files if not path.exists()]

    if missing_downloads:
        raise RuntimeError(
            f"Some downloaded files are missing: {missing_downloads[:10]}"
        )

    print()
    print("Download complete.")
    print(f"Downloaded images: {len(downloaded_files)}")
    print(f"Location: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
