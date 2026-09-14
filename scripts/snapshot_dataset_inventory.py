import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

REPO_ID = "kmmuleelab/Lung_Pathology_Image_JPG"
INVENTORY_PATH = Path("artifacts/manifests/dataset_image_inventory.csv")
PROVENANCE_PATH = Path("artifacts/manifests/dataset_provenance.json")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main() -> None:
    api = HfApi()
    dataset_info = api.dataset_info(REPO_ID)
    revision = dataset_info.sha

    repo_files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
    )

    image_paths = sorted(
        file_path
        for file_path in repo_files
        if Path(file_path).suffix.lower() in IMAGE_EXTENSIONS
    )

    inventory = pd.DataFrame(
        {
            "wsi_id": [Path(path).stem for path in image_paths],
            "repo_path": image_paths,
        }
    )

    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(INVENTORY_PATH, index=False)

    provenance = {
        "repo_id": REPO_ID,
        "repo_type": "dataset",
        "revision": revision,
        "total_repository_files": len(repo_files),
        "image_files": len(image_paths),
    }

    with PROVENANCE_PATH.open("w", encoding="utf-8") as file:
        json.dump(provenance, file, indent=2)
        file.write("\n")

    print("=== Dataset snapshot saved ===")
    print(f"Repository: {REPO_ID}")
    print(f"Revision: {revision}")
    print(f"Repository files: {len(repo_files)}")
    print(f"Image files: {len(image_paths)}")
    print(f"Inventory: {INVENTORY_PATH}")
    print(f"Provenance: {PROVENANCE_PATH}")


if __name__ == "__main__":
    main()
