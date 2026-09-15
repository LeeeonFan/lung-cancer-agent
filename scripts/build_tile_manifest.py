from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import yaml

from lung_fusion_agent.data.tissue import (
    TissueMaskConfig,
    calculate_tissue_mask,
)

CONFIG_PATH = Path("conf/tiling.yaml")
SPLITS_PATH = Path("artifacts/splits/patient_splits.csv")
DOWNSAMPLE_MANIFEST_PATH = Path("artifacts/manifests/downsample_manifest.csv")

OUTPUT_PATH = Path("artifacts/manifests/tile_manifest.csv.gz")
SUMMARY_PATH = Path("artifacts/manifests/tile_summary.csv")


def load_rgb_thumbnail(
    image_path: Path,
    downsample: int,
) -> tuple[np.ndarray, int, int]:
    image = pyvips.Image.new_from_file(
        str(image_path),
        access="sequential",
    )

    thumbnail = image.resize(
        1.0 / downsample,
        kernel="lanczos3",
    )

    if thumbnail.format != "uchar":
        thumbnail = thumbnail.cast("uchar")

    memory = thumbnail.write_to_memory()

    array = np.frombuffer(
        memory,
        dtype=np.uint8,
    ).reshape(
        thumbnail.height,
        thumbnail.width,
        thumbnail.bands,
    )

    if thumbnail.bands != 3:
        raise ValueError(f"Expected three bands: {image_path}")

    return array, image.width, image.height


def integral_mask(mask: np.ndarray) -> np.ndarray:
    return (
        np.pad(
            mask.astype(np.uint32),
            ((1, 0), (1, 0)),
        )
        .cumsum(axis=0)
        .cumsum(axis=1)
    )


def region_fraction(
    integral: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> float:
    tissue_pixels = (
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    )

    area = (x1 - x0) * (y1 - y0)
    return float(tissue_pixels / area)


def wsi_number(wsi_id: str) -> int:
    return int(wsi_id.split("-")[-1])


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    tile_size = int(config["tile"]["size_pixels"])
    stride = int(config["tile"]["stride_pixels"])
    minimum_fraction = float(config["tile"]["minimum_tissue_fraction"])
    maximum_tiles = int(config["tile"]["maximum_tiles_per_slide"])
    mask_downsample = int(config["tissue_mask"]["mask_downsample_from_20x"])
    random_seed = int(config["sampling"]["seed"])

    mask_config = TissueMaskConfig(
        saturation_threshold=float(config["tissue_mask"]["saturation_threshold"]),
        minimum_value=float(config["tissue_mask"]["minimum_value"]),
        maximum_value=float(config["tissue_mask"]["maximum_value"]),
        minimum_component_fraction=float(
            config["tissue_mask"]["minimum_component_fraction"]
        ),
        morphology_kernel_size=int(config["tissue_mask"]["morphology_kernel_size"]),
        opening_iterations=int(config["tissue_mask"]["opening_iterations"]),
        closing_iterations=int(config["tissue_mask"]["closing_iterations"]),
    )

    splits = pd.read_csv(SPLITS_PATH)
    downsample = pd.read_csv(DOWNSAMPLE_MANIFEST_PATH)

    image_paths = dict(
        zip(
            downsample["wsi_id"],
            downsample["output_path"],
            strict=True,
        )
    )

    slides = splits.sort_values("selected_wsi_number").reset_index(drop=True)

    tile_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for slide_index, row in slides.iterrows():
        wsi_id = row["selected_wsi_id"]
        image_path = Path(image_paths[wsi_id])

        print(f"[{slide_index + 1}/{len(slides)}] Building tiles for {wsi_id}")

        rgb, width, height = load_rgb_thumbnail(
            image_path,
            mask_downsample,
        )
        mask = calculate_tissue_mask(rgb, mask_config)
        integral = integral_mask(mask)

        mask_height, mask_width = mask.shape

        candidates: list[tuple[int, int, float]] = []

        for y in range(0, height - tile_size + 1, stride):
            for x in range(
                0,
                width - tile_size + 1,
                stride,
            ):
                mask_x0 = int(np.floor(x * mask_width / width))
                mask_y0 = int(np.floor(y * mask_height / height))
                mask_x1 = int(np.ceil((x + tile_size) * mask_width / width))
                mask_y1 = int(np.ceil((y + tile_size) * mask_height / height))

                mask_x0 = max(0, min(mask_x0, mask_width - 1))
                mask_y0 = max(0, min(mask_y0, mask_height - 1))
                mask_x1 = max(mask_x0 + 1, min(mask_x1, mask_width))
                mask_y1 = max(
                    mask_y0 + 1,
                    min(mask_y1, mask_height),
                )

                tissue_fraction = region_fraction(
                    integral,
                    mask_x0,
                    mask_y0,
                    mask_x1,
                    mask_y1,
                )

                if tissue_fraction >= minimum_fraction:
                    candidates.append((x, y, tissue_fraction))

        candidate_count = len(candidates)

        if candidate_count > maximum_tiles:
            rng = np.random.default_rng(random_seed + wsi_number(wsi_id))
            chosen_indices = rng.choice(
                candidate_count,
                size=maximum_tiles,
                replace=False,
            )
            selected = [candidates[index] for index in sorted(chosen_indices)]
        else:
            selected = candidates

        selected = sorted(
            selected,
            key=lambda item: (item[1], item[0]),
        )

        for rank, (x, y, tissue_fraction) in enumerate(selected):
            tile_records.append(
                {
                    "patient_id": row["patient_id"],
                    "wsi_id": wsi_id,
                    "split": row["split"],
                    "tile_id": (f"{wsi_id}_x{x}_y{y}"),
                    "x": x,
                    "y": y,
                    "tile_size": tile_size,
                    "tissue_fraction": tissue_fraction,
                    "selection_rank": rank,
                }
            )

        summary_records.append(
            {
                "patient_id": row["patient_id"],
                "wsi_id": wsi_id,
                "split": row["split"],
                "width_20x": width,
                "height_20x": height,
                "mask_width": mask_width,
                "mask_height": mask_height,
                "mask_tissue_fraction": float(mask.mean()),
                "candidate_tiles": candidate_count,
                "selected_tiles": len(selected),
                "was_capped": candidate_count > maximum_tiles,
            }
        )

        print(f"  Candidates: {candidate_count}, selected: {len(selected)}")

    tile_manifest = pd.DataFrame(tile_records)
    summary = pd.DataFrame(summary_records)

    assert len(summary) == 204
    assert summary["wsi_id"].is_unique
    assert (summary["selected_tiles"] > 0).all()
    assert (summary["selected_tiles"] <= maximum_tiles).all()
    assert tile_manifest["tile_id"].is_unique
    assert (tile_manifest["tissue_fraction"] >= minimum_fraction).all()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tile_manifest.to_csv(
        OUTPUT_PATH,
        index=False,
        compression={
            "method": "gzip",
            "compresslevel": 9,
            "mtime": 0,
        },
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    print()
    print("=== Tile manifest complete ===")
    print(f"Slides: {len(summary)}")
    print(f"Selected tiles: {len(tile_manifest)}")
    print(f"Capped slides: {summary['was_capped'].sum()}")
    print()
    print("=== Selected tiles per slide ===")
    print(summary["selected_tiles"].describe().to_string())
    print()
    print(f"Tile manifest: {OUTPUT_PATH}")
    print(f"Slide summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
