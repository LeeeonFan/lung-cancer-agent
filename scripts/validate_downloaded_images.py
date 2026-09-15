from pathlib import Path

import pandas as pd
import pyvips

MANIFEST_PATH = Path("artifacts/manifests/patient_manifest.csv")
IMAGE_DIRECTORY = Path("data/raw/images")
OUTPUT_PATH = Path("artifacts/manifests/image_validation.csv")


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)

    records: list[dict[str, object]] = []
    errors: list[str] = []

    for index, row in manifest.iterrows():
        wsi_id = row["selected_wsi_id"]
        repo_path = row["repo_path"]
        image_path = IMAGE_DIRECTORY / repo_path

        print(f"[{index + 1}/{len(manifest)}] Checking {wsi_id}")

        if not image_path.exists():
            errors.append(f"{wsi_id}: file is missing")
            continue

        if image_path.stat().st_size == 0:
            errors.append(f"{wsi_id}: file is empty")
            continue

        try:
            image = pyvips.Image.new_from_file(
                str(image_path),
                access="sequential",
            )
        except pyvips.Error as exception:
            errors.append(f"{wsi_id}: {exception}")
            continue

        if image.width <= 0 or image.height <= 0:
            errors.append(f"{wsi_id}: invalid dimensions {image.width}x{image.height}")
            continue

        if image.bands != 3:
            errors.append(f"{wsi_id}: expected 3 bands, found {image.bands}")
            continue

        records.append(
            {
                "patient_id": row["patient_id"],
                "wsi_id": wsi_id,
                "repo_path": repo_path,
                "file_size_bytes": image_path.stat().st_size,
                "width_80x": image.width,
                "height_80x": image.height,
                "bands": image.bands,
                "pixel_format": image.format,
            }
        )

    validation = pd.DataFrame(records)

    print()
    print("=== Validation summary ===")
    print(f"Expected images: {len(manifest)}")
    print(f"Successfully opened: {len(validation)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print()
        print("=== Errors ===")
        for error in errors:
            print(error)

        raise RuntimeError(f"{len(errors)} images failed validation")

    assert len(validation) == 204
    assert validation["wsi_id"].is_unique
    assert (validation["bands"] == 3).all()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=== Image dimensions ===")
    print(
        validation[["width_80x", "height_80x", "file_size_bytes"]]
        .describe()
        .to_string()
    )
    print()
    print(f"Saved validation table: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
