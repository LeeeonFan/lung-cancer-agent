import json
from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDING_DIR = Path("data/processed/slide_embeddings")
MANIFEST_PATH = Path("artifacts/manifests/slide_embedding_manifest.csv")
SPLIT_PATH = Path("artifacts/splits/patient_splits.csv")
OUTPUT_PATH = Path("artifacts/manifests/slide_embedding_validation.json")

MODEL_FILES = {
    "uni2": (
        "uni2_slide_embeddings.npz",
        1536,
    ),
    "virchow2": (
        "virchow2_slide_embeddings.npz",
        2560,
    ),
    "prism2": (
        "prism2_slide_embeddings.npz",
        2560,
    ),
}


def main() -> None:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype={
            "patient_id": str,
            "selected_wsi_id": str,
        },
    )
    splits = pd.read_csv(
        SPLIT_PATH,
        dtype={
            "patient_id": str,
            "selected_wsi_id": str,
            "grouping_id": str,
        },
    )

    if len(manifest) != 204:
        raise ValueError(f"Expected 204 manifest rows, found {len(manifest)}")

    if not manifest["row_index"].is_unique:
        raise ValueError("Manifest row_index is not unique.")

    if not manifest["patient_id"].is_unique:
        raise ValueError("Manifest patient_id is not unique.")

    if not manifest["selected_wsi_id"].is_unique:
        raise ValueError("Manifest selected_wsi_id is not unique.")

    expected_patient_ids = manifest["patient_id"].astype(str).to_numpy()
    expected_wsi_ids = manifest["selected_wsi_id"].astype(str).to_numpy()
    expected_row_indices = manifest["row_index"].to_numpy(dtype=np.int32)

    validation = {
        "patient_count": len(manifest),
        "tile_count": int(manifest["tile_count"].sum()),
        "models": {},
    }

    for model_name, (
        filename,
        expected_dimension,
    ) in MODEL_FILES.items():
        path = EMBEDDING_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing embedding file: {path}")

        with np.load(
            path,
            allow_pickle=False,
        ) as cache:
            row_indices = cache["row_indices"]
            patient_ids = cache["patient_ids"].astype(str)
            wsi_ids = cache["wsi_ids"].astype(str)
            embeddings = cache["embeddings"]

            expected_shape = (
                204,
                expected_dimension,
            )

            if embeddings.shape != expected_shape:
                raise ValueError(
                    f"{model_name}: shape {embeddings.shape}, expected {expected_shape}"
                )

            if embeddings.dtype != np.float32:
                raise ValueError(
                    f"{model_name}: dtype {embeddings.dtype}, expected float32"
                )

            if not np.isfinite(embeddings).all():
                raise ValueError(f"{model_name}: contains NaN or infinity")

            if not np.array_equal(
                row_indices,
                expected_row_indices,
            ):
                raise ValueError(f"{model_name}: row-index order mismatch")

            if not np.array_equal(
                patient_ids,
                expected_patient_ids,
            ):
                raise ValueError(f"{model_name}: patient-ID order mismatch")

            if not np.array_equal(
                wsi_ids,
                expected_wsi_ids,
            ):
                raise ValueError(f"{model_name}: WSI-ID order mismatch")

            norms = np.linalg.norm(
                embeddings,
                axis=1,
            )

            validation["models"][model_name] = {
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "all_finite": True,
                "row_order_matches": True,
                "patient_order_matches": True,
                "wsi_order_matches": True,
                "mean_l2_norm": float(norms.mean()),
                "min_l2_norm": float(norms.min()),
                "max_l2_norm": float(norms.max()),
            }

    if len(splits) != 204:
        raise ValueError(f"Expected 204 split rows, found {len(splits)}")

    if set(splits["patient_id"]) != set(manifest["patient_id"]):
        raise ValueError("Split and embedding patient IDs do not match.")

    grouping_split_counts = splits.groupby("grouping_id")["split"].nunique()

    leaking_groups = grouping_split_counts[grouping_split_counts > 1]

    if not leaking_groups.empty:
        raise ValueError(
            f"Found {len(leaking_groups)} grouping IDs across multiple splits."
        )

    split_counts = splits["split"].value_counts().to_dict()

    expected_split_counts = {
        "train": 141,
        "validation": 21,
        "test": 42,
    }

    if split_counts != expected_split_counts:
        raise ValueError(f"Unexpected split counts: {split_counts}")

    validation["split_counts"] = split_counts
    validation["group_leakage_detected"] = False

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(
            validation,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("=== Slide embedding validation ===")
    print("Patients/slides:", len(manifest))
    print(
        "Tiles:",
        int(manifest["tile_count"].sum()),
    )

    for model_name, result in validation["models"].items():
        print(
            f"{model_name}: "
            f"shape={tuple(result['shape'])}, "
            f"dtype={result['dtype']}, "
            f"finite={result['all_finite']}"
        )

    print("Split counts:", split_counts)
    print(
        "Grouping leakage detected:",
        False,
    )
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
