from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

SELECTION_PATH = Path("artifacts/manifests/qc_train_selection.csv")
OUTPUT_DIRECTORY = Path("data/processed/qc_train/tissue_masks")

SATURATION_THRESHOLD = 0.08
MIN_VALUE = 0.20
MAX_VALUE = 0.97
MIN_COMPONENT_FRACTION = 0.0005


def calculate_tissue_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)

    saturation = (maximum - minimum) / (maximum + 1e-6)
    value = maximum

    mask = (
        (saturation >= SATURATION_THRESHOLD)
        & (value >= MIN_VALUE)
        & (value <= MAX_VALUE)
    )

    structure = np.ones((3, 3), dtype=bool)

    mask = ndimage.binary_opening(
        mask,
        structure=structure,
        iterations=1,
    )
    mask = ndimage.binary_closing(
        mask,
        structure=structure,
        iterations=2,
    )

    labels, component_count = ndimage.label(mask)

    if component_count > 0:
        component_sizes = np.bincount(labels.ravel())
        minimum_size = max(
            64,
            int(mask.size * MIN_COMPONENT_FRACTION),
        )

        keep = component_sizes >= minimum_size
        keep[0] = False
        mask = keep[labels]

    return mask


def create_overlay(
    image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    original = image.convert("RGB")
    original_array = np.asarray(
        original,
        dtype=np.float32,
    )

    overlay_array = original_array.copy()

    green = np.array(
        [30.0, 220.0, 60.0],
        dtype=np.float32,
    )
    alpha = 0.38

    overlay_array[mask] = (1.0 - alpha) * overlay_array[mask] + alpha * green

    return Image.fromarray(np.clip(overlay_array, 0, 255).astype(np.uint8))


def main() -> None:
    selection = pd.read_csv(SELECTION_PATH)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []

    for index, row in selection.iterrows():
        thumbnail_path = Path(row["thumbnail_path"])
        wsi_id = row["wsi_id"]
        label = row["label"]

        print(f"[{index + 1}/{len(selection)}] {label}: {wsi_id}")

        with Image.open(thumbnail_path) as source:
            image = source.convert("RGB")

        mask = calculate_tissue_mask(image)
        overlay = create_overlay(image, mask)

        mask_path = OUTPUT_DIRECTORY / f"{wsi_id}_mask.png"
        overlay_path = OUTPUT_DIRECTORY / f"{wsi_id}_overlay.jpg"

        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

        overlay.save(
            overlay_path,
            quality=92,
        )

        tissue_fraction = float(mask.mean())

        records.append(
            {
                "patient_id": row["patient_id"],
                "wsi_id": wsi_id,
                "label": label,
                "tissue_fraction": tissue_fraction,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )

        print(f"  Tissue fraction: {tissue_fraction:.1%}")

    results = pd.DataFrame(records)

    results.to_csv(
        OUTPUT_DIRECTORY / "tissue_mask_qc.csv",
        index=False,
    )

    print()
    print("=== Tissue-mask QC complete ===")
    print(results[["label", "wsi_id", "tissue_fraction"]].to_string(index=False))
    print()
    print(f"Outputs: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
