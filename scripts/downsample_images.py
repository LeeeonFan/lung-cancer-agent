import time
from pathlib import Path

import pandas as pd
import pyvips

SCALE = 0.25
JPEG_QUALITY = 95
KERNEL = "lanczos3"

VALIDATION_PATH = Path("artifacts/manifests/image_validation.csv")
INPUT_DIRECTORY = Path("data/raw/images")
OUTPUT_DIRECTORY = Path("data/processed/images_20x")
OUTPUT_MANIFEST_PATH = Path("artifacts/manifests/downsample_manifest.csv")


def expected_dimension(original_dimension: int) -> int:
    """Return the expected dimension after 4x linear downsampling."""
    return int(original_dimension * SCALE + 0.5)


def validate_output(
    output_path: Path,
    expected_width: int,
    expected_height: int,
) -> tuple[bool, int, int, int]:
    """Check that an existing output has the expected image header."""
    try:
        image = pyvips.Image.new_from_file(str(output_path))
    except pyvips.Error:
        return False, 0, 0, 0

    valid = (
        image.width == expected_width
        and image.height == expected_height
        and image.bands == 3
    )

    return valid, image.width, image.height, image.bands


def main() -> None:
    validation = pd.read_csv(VALIDATION_PATH)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    total_start = time.perf_counter()

    for index, row in validation.iterrows():
        wsi_id = row["wsi_id"]
        repo_path = row["repo_path"]

        input_path = INPUT_DIRECTORY / repo_path
        output_path = OUTPUT_DIRECTORY / repo_path
        temporary_path = output_path.with_suffix(".partial.jpg")

        expected_width = expected_dimension(int(row["width_80x"]))
        expected_height = expected_dimension(int(row["height_80x"]))

        print(f"[{index + 1}/{len(validation)}] Processing {wsi_id}")

        if output_path.exists():
            valid, width, height, bands = validate_output(
                output_path,
                expected_width,
                expected_height,
            )

            if valid:
                print("  Existing valid output found; skipping.")

                records.append(
                    {
                        "patient_id": row["patient_id"],
                        "wsi_id": wsi_id,
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "width_80x": int(row["width_80x"]),
                        "height_80x": int(row["height_80x"]),
                        "width_20x": width,
                        "height_20x": height,
                        "bands": bands,
                        "output_size_bytes": (output_path.stat().st_size),
                        "scale": SCALE,
                        "kernel": KERNEL,
                        "jpeg_quality": JPEG_QUALITY,
                        "status": "existing_valid",
                        "processing_seconds": 0.0,
                    }
                )
                continue

            raise RuntimeError(f"Existing output is invalid: {output_path}")

        # A previous interrupted run may leave a partial file.
        if temporary_path.exists():
            temporary_path.unlink()

        slide_start = time.perf_counter()

        try:
            source = pyvips.Image.new_from_file(
                str(input_path),
                access="sequential",
            )
            output = source.resize(
                SCALE,
                kernel=KERNEL,
            )
            output.jpegsave(
                str(temporary_path),
                Q=JPEG_QUALITY,
                optimize_coding=True,
            )
        except (pyvips.Error, OSError) as exception:
            if temporary_path.exists():
                temporary_path.unlink()

            raise RuntimeError(f"Failed to downsample {wsi_id}") from exception

        valid, width, height, bands = validate_output(
            temporary_path,
            expected_width,
            expected_height,
        )

        if not valid:
            if temporary_path.exists():
                temporary_path.unlink()

            raise RuntimeError(
                f"Invalid downsampled output for {wsi_id}: "
                f"{width}x{height}, bands={bands}"
            )

        # Rename only after successful processing and validation.
        temporary_path.replace(output_path)

        elapsed = time.perf_counter() - slide_start
        print(f"  Saved {width}x{height} in {elapsed:.1f} seconds.")

        records.append(
            {
                "patient_id": row["patient_id"],
                "wsi_id": wsi_id,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "width_80x": int(row["width_80x"]),
                "height_80x": int(row["height_80x"]),
                "width_20x": width,
                "height_20x": height,
                "bands": bands,
                "output_size_bytes": output_path.stat().st_size,
                "scale": SCALE,
                "kernel": KERNEL,
                "jpeg_quality": JPEG_QUALITY,
                "status": "created",
                "processing_seconds": elapsed,
            }
        )

    output_manifest = pd.DataFrame(records)

    assert len(output_manifest) == len(validation) == 204
    assert output_manifest["wsi_id"].is_unique
    assert (output_manifest["bands"] == 3).all()

    OUTPUT_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_manifest.to_csv(
        OUTPUT_MANIFEST_PATH,
        index=False,
    )

    total_elapsed = time.perf_counter() - total_start

    print()
    print("=== Downsampling complete ===")
    print(f"Processed images: {len(output_manifest)}")
    print(
        "Total output size: "
        f"{output_manifest['output_size_bytes'].sum() / 1024**3:.2f} GB"
    )
    print(f"Total elapsed time: {total_elapsed / 60:.1f} minutes")
    print(f"Manifest: {OUTPUT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
