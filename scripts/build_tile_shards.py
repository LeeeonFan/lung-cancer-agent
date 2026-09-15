import argparse
import hashlib
import io
import json
import re
import tarfile
import time
from pathlib import Path

import pandas as pd
import pyvips

TILE_MANIFEST_PATH = Path("artifacts/manifests/tile_manifest.csv.gz")
IMAGE_DIR = Path("data/processed/images_20x")
OUTPUT_DIR = Path("data/processed/tile_shards")
SHARD_MANIFEST_PATH = Path("artifacts/manifests/tile_shards.csv")
TILE_INDEX_PATH = Path("artifacts/manifests/tile_to_shard.csv.gz")

DEFAULT_SHARD_SIZE = 4096
JPEG_QUALITY = 95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package selected pathology tiles into deterministic TAR shards."
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Number of tiles per TAR shard.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Only process this many shards. Useful for a pilot run.",
    )
    return parser.parse_args()


def wsi_number(wsi_id: str) -> int:
    match = re.fullmatch(r"WSI-(\d+)", wsi_id)
    if match is None:
        raise ValueError(f"Unexpected WSI ID: {wsi_id}")
    return int(match.group(1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def make_tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def validate_existing_shard(path: Path, expected_tiles: int) -> None:
    with tarfile.open(path, mode="r") as archive:
        jpg_count = sum(
            member.isfile() and member.name.endswith(".jpg")
            for member in archive.getmembers()
        )

    if jpg_count != expected_tiles:
        raise RuntimeError(
            f"Existing shard {path} contains {jpg_count} JPEG tiles; "
            f"expected {expected_tiles}. Delete this shard and rerun."
        )


def build_shard(
    shard_frame: pd.DataFrame,
    shard_path: Path,
) -> None:
    partial_path = shard_path.with_suffix(".partial.tar")

    if partial_path.exists():
        partial_path.unlink()

    current_wsi_id: str | None = None
    current_image: pyvips.Image | None = None

    with tarfile.open(
        partial_path,
        mode="w",
        format=tarfile.GNU_FORMAT,
    ) as archive:
        total_tiles = len(shard_frame)

        for position, row in enumerate(
            shard_frame.itertuples(index=False),
            start=1,
        ):
            if row.wsi_id != current_wsi_id:
                image_path = IMAGE_DIR / f"{row.wsi_id}.jpg"

                if not image_path.exists():
                    raise FileNotFoundError(f"Missing 20x image: {image_path}")

                current_image = pyvips.Image.new_from_file(
                    str(image_path),
                    access="random",
                )
                current_wsi_id = row.wsi_id

            if current_image is None:
                raise RuntimeError("No source image is open.")

            x = int(row.x)
            y = int(row.y)
            tile_size = int(row.tile_size)

            if x + tile_size > current_image.width:
                raise ValueError(f"Tile extends past image width: {row.tile_id}")

            if y + tile_size > current_image.height:
                raise ValueError(f"Tile extends past image height: {row.tile_id}")

            tile = current_image.crop(x, y, tile_size, tile_size)
            jpeg_bytes = tile.write_to_buffer(
                ".jpg",
                Q=JPEG_QUALITY,
                strip=True,
            )

            image_member = f"{row.tile_id}.jpg"
            image_info = make_tar_info(image_member, len(jpeg_bytes))
            archive.addfile(image_info, io.BytesIO(jpeg_bytes))

            metadata = {
                "patient_id": row.patient_id,
                "tile_id": row.tile_id,
                "tile_size": tile_size,
                "tissue_fraction": round(float(row.tissue_fraction), 6),
                "wsi_id": row.wsi_id,
                "x": x,
                "y": y,
            }
            metadata_bytes = json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

            metadata_member = f"{row.tile_id}.json"
            metadata_info = make_tar_info(metadata_member, len(metadata_bytes))
            archive.addfile(metadata_info, io.BytesIO(metadata_bytes))

            if position % 256 == 0 or position == total_tiles:
                print(f"  Packed {position}/{total_tiles} tiles")

    partial_path.replace(shard_path)


def main() -> None:
    args = parse_args()

    if args.shard_size <= 0:
        raise ValueError("--shard-size must be greater than zero.")

    tile_manifest = pd.read_csv(TILE_MANIFEST_PATH)

    required_columns = {
        "patient_id",
        "wsi_id",
        "tile_id",
        "x",
        "y",
        "tile_size",
        "tissue_fraction",
        "selection_rank",
    }
    missing_columns = required_columns - set(tile_manifest.columns)

    if missing_columns:
        raise ValueError(f"Tile manifest is missing columns: {sorted(missing_columns)}")

    tile_manifest["_wsi_number"] = tile_manifest["wsi_id"].map(wsi_number)
    tile_manifest = tile_manifest.sort_values(
        ["_wsi_number", "selection_rank", "tile_id"],
        kind="stable",
    ).reset_index(drop=True)

    total_tiles = len(tile_manifest)
    total_shards = (total_tiles + args.shard_size - 1) // args.shard_size

    if args.max_shards is None:
        shards_to_process = total_shards
    else:
        shards_to_process = min(args.max_shards, total_shards)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=== Tile shard plan ===")
    print(f"Tiles: {total_tiles}")
    print(f"Tiles per shard: {args.shard_size}")
    print(f"Total shards: {total_shards}")
    print(f"Shards processed this run: {shards_to_process}")

    shard_records: list[dict[str, object]] = []
    tile_index_records: list[dict[str, object]] = []
    start_time = time.perf_counter()

    for shard_index in range(shards_to_process):
        start = shard_index * args.shard_size
        end = min(start + args.shard_size, total_tiles)
        shard_frame = tile_manifest.iloc[start:end].copy()

        shard_name = f"tiles-{shard_index:05d}.tar"
        shard_path = OUTPUT_DIR / shard_name
        expected_tiles = len(shard_frame)

        print(
            f"\n[{shard_index + 1}/{shards_to_process}] "
            f"{shard_name}: {expected_tiles} tiles"
        )

        shard_start = time.perf_counter()

        if shard_path.exists():
            print("  Existing shard found; validating...")
            validate_existing_shard(shard_path, expected_tiles)
            print("  Existing shard is valid; skipping rebuild.")
        else:
            build_shard(shard_frame, shard_path)

        checksum = sha256_file(shard_path)
        elapsed = time.perf_counter() - shard_start

        shard_records.append(
            {
                "shard_index": shard_index,
                "shard_name": shard_name,
                "tile_count": expected_tiles,
                "first_tile_id": shard_frame.iloc[0]["tile_id"],
                "last_tile_id": shard_frame.iloc[-1]["tile_id"],
                "size_bytes": shard_path.stat().st_size,
                "sha256": checksum,
                "elapsed_seconds": round(elapsed, 3),
            }
        )

        for row in shard_frame.itertuples(index=False):
            tile_index_records.append(
                {
                    "tile_id": row.tile_id,
                    "wsi_id": row.wsi_id,
                    "patient_id": row.patient_id,
                    "shard_name": shard_name,
                    "image_member": f"{row.tile_id}.jpg",
                    "metadata_member": f"{row.tile_id}.json",
                }
            )

        size_gb = shard_path.stat().st_size / 1_000_000_000
        print(f"  Size: {size_gb:.2f} GB")
        print(f"  SHA-256: {checksum}")
        print(f"  Time: {elapsed / 60:.1f} minutes")

    shard_manifest = pd.DataFrame(shard_records)
    tile_index = pd.DataFrame(tile_index_records)

    shard_manifest.to_csv(SHARD_MANIFEST_PATH, index=False)
    tile_index.to_csv(
        TILE_INDEX_PATH,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    elapsed_total = time.perf_counter() - start_time

    print("\n=== Sharding complete ===")
    print(f"Completed shards: {len(shard_manifest)}")
    print(f"Indexed tiles: {len(tile_index)}")
    print(
        "Total output size: "
        f"{shard_manifest['size_bytes'].sum() / 1_000_000_000:.2f} GB"
    )
    print(f"Total elapsed time: {elapsed_total / 60:.1f} minutes")
    print(f"Shard manifest: {SHARD_MANIFEST_PATH}")
    print(f"Tile index: {TILE_INDEX_PATH}")

    if shards_to_process < total_shards:
        print(
            "\nPilot run only. Run again without --max-shards "
            "to build all remaining shards."
        )


if __name__ == "__main__":
    main()
