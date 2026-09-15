import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PATIENT_MANIFEST_PATH = Path("artifacts/manifests/patient_manifest.csv")
TILE_INDEX_PATH = Path("artifacts/manifests/tile_to_shard.csv.gz")

EMBEDDING_ROOT = Path("data/processed/embeddings")
OUTPUT_DIR = Path("data/processed/slide_embeddings")

OUTPUT_MANIFEST_PATH = Path("artifacts/manifests/slide_embedding_manifest.csv")
PROVENANCE_PATH = Path("artifacts/manifests/slide_embedding_provenance.json")

EMBEDDING_REPOSITORY = "AliothMe/lung-fusion-embeddings"
EMBEDDING_REPOSITORY_REVISION = "06bdbaa228c08813e4a87704b300c5c9335d7e58"

MODEL_CONFIGS = {
    "uni2": {
        "input_directory": EMBEDDING_ROOT / "uni2",
        "pattern": "uni2-*.npz",
        "expected_files": 49,
        "tile_dimension": 1536,
        "output_filename": "uni2_slide_embeddings.npz",
        "source_model": "MahmoodLab/UNI2-h",
        "source_revision": ("d517a8dd47902dd7c308b3c36f63bce47e7b9a43"),
    },
    "virchow2": {
        "input_directory": EMBEDDING_ROOT / "virchow2",
        "pattern": "virchow2-*.npz",
        "expected_files": 49,
        "tile_dimension": 2560,
        "output_filename": "virchow2_slide_embeddings.npz",
        "source_model": "paige-ai/Virchow2",
        "source_revision": ("3158645804b69e3f3bc4439d4116edddf0840a72"),
    },
}


def wsi_number(wsi_id: str) -> int:
    prefix, number = wsi_id.split("-", maxsplit=1)

    if prefix != "WSI":
        raise ValueError(f"Unexpected WSI ID: {wsi_id}")

    return int(number)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def validate_patient_manifest(
    patient_manifest: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "patient_id",
        "selected_wsi_id",
    }
    missing_columns = required_columns - set(patient_manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Patient manifest is missing columns: {sorted(missing_columns)}"
        )

    if len(patient_manifest) != 204:
        raise ValueError(f"Expected 204 patient records, found {len(patient_manifest)}")

    if not patient_manifest["patient_id"].is_unique:
        raise ValueError("patient_id is not unique.")

    if not patient_manifest["selected_wsi_id"].is_unique:
        raise ValueError("selected_wsi_id is not unique.")

    result = patient_manifest[["patient_id", "selected_wsi_id"]].copy()

    result["_wsi_number"] = result["selected_wsi_id"].map(wsi_number)

    result = (
        result.sort_values(
            "_wsi_number",
            kind="stable",
        )
        .drop(columns="_wsi_number")
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "row_index",
        range(len(result)),
    )

    return result


def validate_tile_index(
    tile_index: pd.DataFrame,
    canonical_wsi_ids: list[str],
) -> None:
    required_columns = {
        "tile_id",
        "wsi_id",
        "shard_name",
    }
    missing_columns = required_columns - set(tile_index.columns)

    if missing_columns:
        raise ValueError(f"Tile index is missing columns: {sorted(missing_columns)}")

    if len(tile_index) != 200488:
        raise ValueError(f"Expected 200488 tiles, found {len(tile_index)}")

    if not tile_index["tile_id"].is_unique:
        raise ValueError("tile_id is not unique.")

    observed_wsi_ids = set(tile_index["wsi_id"].astype(str))
    expected_wsi_ids = set(canonical_wsi_ids)

    if observed_wsi_ids != expected_wsi_ids:
        raise ValueError("Tile-index WSI IDs do not match patient manifest.")


def aggregate_tile_embeddings(
    *,
    model_name: str,
    config: dict[str, object],
    tile_index: pd.DataFrame,
    canonical_table: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    input_directory = Path(config["input_directory"])
    embedding_files = sorted(input_directory.glob(str(config["pattern"])))

    expected_files = int(config["expected_files"])
    expected_dimension = int(config["tile_dimension"])

    if len(embedding_files) != expected_files:
        raise ValueError(
            f"{model_name}: expected "
            f"{expected_files} NPZ files, "
            f"found {len(embedding_files)}"
        )

    tile_to_wsi = tile_index.set_index("tile_id")["wsi_id"].astype(str)

    expected_tile_ids = set(tile_index["tile_id"].astype(str))
    observed_tile_ids: set[str] = set()

    embedding_sums: dict[str, np.ndarray] = {}
    embedding_counts: dict[str, int] = defaultdict(int)

    for file_number, embedding_path in enumerate(
        embedding_files,
        start=1,
    ):
        with np.load(
            embedding_path,
            allow_pickle=False,
        ) as cache:
            tile_ids = cache["tile_ids"].astype(str)
            embeddings = cache["embeddings"]

            if embeddings.ndim != 2:
                raise ValueError(
                    f"{embedding_path}: embeddings must be two-dimensional."
                )

            if embeddings.shape != (
                len(tile_ids),
                expected_dimension,
            ):
                raise ValueError(
                    f"{embedding_path}: unexpected shape {embeddings.shape}"
                )

            if not np.isfinite(embeddings).all():
                raise ValueError(
                    f"{embedding_path}: embeddings contain NaN or infinity."
                )

            current_ids = set(tile_ids.tolist())
            overlap = observed_tile_ids & current_ids

            if overlap:
                raise ValueError(
                    f"{embedding_path}: found {len(overlap)} duplicate tile IDs."
                )

            unknown_ids = current_ids - expected_tile_ids

            if unknown_ids:
                raise ValueError(
                    f"{embedding_path}: found {len(unknown_ids)} unknown tile IDs."
                )

            observed_tile_ids.update(current_ids)

            wsi_ids = tile_to_wsi.loc[tile_ids].to_numpy()

            embeddings_float32 = embeddings.astype(np.float32)

            for wsi_id in pd.unique(wsi_ids):
                mask = wsi_ids == wsi_id
                selected = embeddings_float32[mask]

                selected_sum = selected.sum(
                    axis=0,
                    dtype=np.float64,
                )

                if wsi_id not in embedding_sums:
                    embedding_sums[wsi_id] = np.zeros(
                        expected_dimension,
                        dtype=np.float64,
                    )

                embedding_sums[wsi_id] += selected_sum
                embedding_counts[wsi_id] += len(selected)

        print(
            f"{model_name}: "
            f"[{file_number:02d}/"
            f"{len(embedding_files)}] "
            f"{embedding_path.name}"
        )

    missing_tile_ids = expected_tile_ids - observed_tile_ids

    if missing_tile_ids:
        raise ValueError(
            f"{model_name}: missing {len(missing_tile_ids)} tile embeddings."
        )

    expected_counts = tile_index.groupby("wsi_id").size().to_dict()

    slide_embeddings = []

    for wsi_id in canonical_table["selected_wsi_id"]:
        observed_count = embedding_counts[wsi_id]
        expected_count = int(expected_counts[wsi_id])

        if observed_count != expected_count:
            raise ValueError(
                f"{model_name}/{wsi_id}: "
                f"expected {expected_count} "
                f"tiles, found {observed_count}"
            )

        mean_embedding = (embedding_sums[wsi_id] / observed_count).astype(np.float32)

        if not np.isfinite(mean_embedding).all():
            raise ValueError(f"{model_name}/{wsi_id}: non-finite slide embedding.")

        slide_embeddings.append(mean_embedding)

    matrix = np.stack(
        slide_embeddings,
        axis=0,
    )

    expected_shape = (
        len(canonical_table),
        expected_dimension,
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"{model_name}: final shape {matrix.shape}, expected {expected_shape}"
        )

    metadata = {
        "model_name": model_name,
        "source_model": config["source_model"],
        "source_revision": config["source_revision"],
        "source_file_count": len(embedding_files),
        "source_tile_count": len(observed_tile_ids),
        "tile_embedding_dimension": (expected_dimension),
        "slide_embedding_dimension": (expected_dimension),
        "slide_count": len(canonical_table),
        "pooling": "arithmetic_mean",
        "accumulation_dtype": "float64",
        "output_dtype": "float32",
    }

    return matrix, metadata


def load_prism2_embeddings(
    canonical_table: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    prism2_path = EMBEDDING_ROOT / "prism2" / "prism2_base_embeddings.npz"

    if not prism2_path.exists():
        raise FileNotFoundError(f"Missing Prism2 embeddings: {prism2_path}")

    with np.load(
        prism2_path,
        allow_pickle=False,
    ) as cache:
        wsi_ids = cache["wsi_ids"].astype(str)
        embeddings = cache["embeddings"].astype(np.float32)
        tile_counts = cache["tile_counts"].astype(np.int32)

    if embeddings.shape != (
        204,
        2560,
    ):
        raise ValueError(f"Unexpected Prism2 shape: {embeddings.shape}")

    if len(set(wsi_ids.tolist())) != 204:
        raise ValueError("Prism2 WSI IDs are not unique.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Prism2 embeddings contain NaN or infinity.")

    row_by_wsi = {wsi_id: row_index for row_index, wsi_id in enumerate(wsi_ids)}

    canonical_wsi_ids = canonical_table["selected_wsi_id"].tolist()

    missing_wsi_ids = set(canonical_wsi_ids) - set(row_by_wsi)

    if missing_wsi_ids:
        raise ValueError(f"Prism2 is missing {len(missing_wsi_ids)} WSIs.")

    reorder_indices = [row_by_wsi[wsi_id] for wsi_id in canonical_wsi_ids]

    reordered_embeddings = embeddings[reorder_indices]
    reordered_tile_counts = tile_counts[reorder_indices]

    metadata = {
        "model_name": "prism2",
        "source_model": "paige-ai/Prism2",
        "source_revision": ("450352d0ddc6b42b21ce20794ce0fbefe6b5a47a"),
        "input_model": "paige-ai/Virchow2",
        "input_feature": ("class_token_only"),
        "input_dimension": 1280,
        "source_tile_count": int(reordered_tile_counts.sum()),
        "slide_count": len(canonical_table),
        "slide_embedding_dimension": 2560,
        "pooling": "prism2_perceiver_base",
        "output_dtype": "float32",
    }

    return reordered_embeddings, metadata


def save_slide_embeddings(
    *,
    model_name: str,
    output_filename: str,
    embeddings: np.ndarray,
    canonical_table: pd.DataFrame,
) -> tuple[Path, str]:
    output_path = OUTPUT_DIR / output_filename

    np.savez_compressed(
        output_path,
        row_indices=canonical_table["row_index"].to_numpy(dtype=np.int32),
        patient_ids=np.asarray(
            canonical_table["patient_id"].astype(str).tolist(),
            dtype=np.str_,
        ),
        wsi_ids=np.asarray(
            canonical_table["selected_wsi_id"].astype(str).tolist(),
            dtype=np.str_,
        ),
        embeddings=embeddings.astype(np.float32),
    )

    checksum = sha256_file(output_path)

    print(f"{model_name}: saved {output_path}")
    print(f"{model_name}: shape {embeddings.shape}")
    print(f"{model_name}: SHA-256 {checksum}")

    return output_path, checksum


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    PROVENANCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_manifest = pd.read_csv(
        PATIENT_MANIFEST_PATH,
        dtype={
            "patient_id": str,
            "selected_wsi_id": str,
        },
    )
    canonical_table = validate_patient_manifest(patient_manifest)

    tile_index = pd.read_csv(
        TILE_INDEX_PATH,
        dtype={
            "tile_id": str,
            "wsi_id": str,
        },
    )

    canonical_wsi_ids = canonical_table["selected_wsi_id"].tolist()

    validate_tile_index(
        tile_index,
        canonical_wsi_ids,
    )

    tile_counts = tile_index.groupby("wsi_id").size().to_dict()

    output_manifest = canonical_table.copy()
    output_manifest["tile_count"] = (
        output_manifest["selected_wsi_id"].map(tile_counts).astype(int)
    )

    provenance = {
        "embedding_repository": (EMBEDDING_REPOSITORY),
        "embedding_repository_revision": (EMBEDDING_REPOSITORY_REVISION),
        "canonical_order": ("numeric selected_wsi_id"),
        "patient_count": len(canonical_table),
        "tile_count": len(tile_index),
        "models": {},
    }

    for model_name, config in MODEL_CONFIGS.items():
        matrix, model_metadata = aggregate_tile_embeddings(
            model_name=model_name,
            config=config,
            tile_index=tile_index,
            canonical_table=canonical_table,
        )

        output_path, checksum = save_slide_embeddings(
            model_name=model_name,
            output_filename=str(config["output_filename"]),
            embeddings=matrix,
            canonical_table=canonical_table,
        )

        model_metadata["output_path"] = str(output_path)
        model_metadata["output_sha256"] = checksum

        provenance["models"][model_name] = model_metadata

    prism2_matrix, prism2_metadata = load_prism2_embeddings(canonical_table)

    prism2_output_path, prism2_checksum = save_slide_embeddings(
        model_name="prism2",
        output_filename=("prism2_slide_embeddings.npz"),
        embeddings=prism2_matrix,
        canonical_table=canonical_table,
    )

    prism2_metadata["output_path"] = str(prism2_output_path)
    prism2_metadata["output_sha256"] = prism2_checksum

    provenance["models"]["prism2"] = prism2_metadata

    output_manifest.to_csv(
        OUTPUT_MANIFEST_PATH,
        index=False,
    )

    PROVENANCE_PATH.write_text(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=== Slide embeddings complete ===")
    print(
        "Patients/slides:",
        len(canonical_table),
    )
    print("Tiles:", len(tile_index))
    print(
        "UNI2 shape:",
        provenance["models"]["uni2"]["slide_count"],
        "x",
        provenance["models"]["uni2"]["slide_embedding_dimension"],
    )
    print(
        "Virchow2 shape:",
        provenance["models"]["virchow2"]["slide_count"],
        "x",
        provenance["models"]["virchow2"]["slide_embedding_dimension"],
    )
    print(
        "Prism2 shape:",
        provenance["models"]["prism2"]["slide_count"],
        "x",
        provenance["models"]["prism2"]["slide_embedding_dimension"],
    )
    print(
        "Manifest:",
        OUTPUT_MANIFEST_PATH,
    )
    print(
        "Provenance:",
        PROVENANCE_PATH,
    )


if __name__ == "__main__":
    main()
