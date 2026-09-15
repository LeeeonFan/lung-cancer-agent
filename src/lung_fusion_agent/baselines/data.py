from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_PATH = Path("artifacts/splits/patient_splits.csv")
EMBEDDING_DIR = Path("data/processed/slide_embeddings")

CLASS_NAMES = [
    "Acinar",
    "Cribriform",
    "In situ",
    "Lepidic",
    "Micropapillary",
    "Papillary",
    "Solid",
]

MODEL_FILES = {
    "uni2": (EMBEDDING_DIR / "uni2_slide_embeddings.npz"),
    "virchow2": (EMBEDDING_DIR / "virchow2_slide_embeddings.npz"),
    "prism2": (EMBEDDING_DIR / "prism2_slide_embeddings.npz"),
}


@dataclass(frozen=True)
class BaselineDataset:
    patient_ids: np.ndarray
    grouping_ids: np.ndarray
    wsi_ids: np.ndarray
    labels: np.ndarray
    label_names: np.ndarray
    splits: np.ndarray
    representations: dict[str, np.ndarray]

    def mask_for_split(
        self,
        split_name: str,
    ) -> np.ndarray:
        mask = self.splits == split_name

        if not mask.any():
            raise ValueError(f"No records found for split: {split_name}")

        return mask


def load_embedding_file(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding file: {path}")

    with np.load(
        path,
        allow_pickle=False,
    ) as cache:
        patient_ids = cache["patient_ids"].astype(str)
        wsi_ids = cache["wsi_ids"].astype(str)
        embeddings = cache["embeddings"].astype(np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"{path}: embeddings must be two-dimensional.")

    if len(patient_ids) != len(embeddings):
        raise ValueError(f"{path}: patient count mismatch.")

    if len(wsi_ids) != len(embeddings):
        raise ValueError(f"{path}: WSI count mismatch.")

    if not np.isfinite(embeddings).all():
        raise ValueError(f"{path}: embeddings contain NaN or infinity.")

    return patient_ids, wsi_ids, embeddings


def load_baseline_dataset() -> BaselineDataset:
    splits = pd.read_csv(
        SPLIT_PATH,
        dtype={
            "patient_id": str,
            "grouping_id": str,
            "selected_wsi_id": str,
        },
    )

    required_columns = {
        "patient_id",
        "grouping_id",
        "selected_wsi_id",
        "age",
        "sex",
        "label",
        "split",
    }
    missing_columns = required_columns - set(splits.columns)

    if missing_columns:
        raise ValueError(f"Split file is missing columns: {sorted(missing_columns)}")

    if len(splits) != 204:
        raise ValueError(f"Expected 204 split records, found {len(splits)}")

    if not splits["patient_id"].is_unique:
        raise ValueError("patient_id is not unique.")

    reference_patient_ids = None
    reference_wsi_ids = None
    representations = {}

    for model_name, path in MODEL_FILES.items():
        (
            patient_ids,
            wsi_ids,
            embeddings,
        ) = load_embedding_file(path)

        if reference_patient_ids is None:
            reference_patient_ids = patient_ids
            reference_wsi_ids = wsi_ids
        else:
            if not np.array_equal(
                patient_ids,
                reference_patient_ids,
            ):
                raise ValueError(f"{model_name}: patient order mismatch.")

            if not np.array_equal(
                wsi_ids,
                reference_wsi_ids,
            ):
                raise ValueError(f"{model_name}: WSI order mismatch.")

        representations[model_name] = embeddings

    if reference_patient_ids is None:
        raise RuntimeError("No embedding files were loaded.")

    split_by_patient = splits.set_index("patient_id")

    missing_patient_ids = set(reference_patient_ids) - set(split_by_patient.index)

    if missing_patient_ids:
        raise ValueError(
            f"Split file is missing {len(missing_patient_ids)} embedding patients."
        )

    aligned = split_by_patient.loc[reference_patient_ids].reset_index()

    aligned_wsi_ids = aligned["selected_wsi_id"].astype(str).to_numpy()

    if not np.array_equal(
        aligned_wsi_ids,
        reference_wsi_ids,
    ):
        raise ValueError("Split WSI order does not match embedding WSI order.")

    unexpected_sex_values = set(aligned["sex"]) - {"Female", "Male"}

    if unexpected_sex_values:
        raise ValueError(f"Unexpected sex values: {sorted(unexpected_sex_values)}")

    age = aligned["age"].to_numpy(dtype=np.float32)
    sex_male = (aligned["sex"] == "Male").to_numpy(dtype=np.float32)

    metadata = np.column_stack([age, sex_male]).astype(np.float32)

    label_to_index = {
        class_name: class_index for class_index, class_name in enumerate(CLASS_NAMES)
    }

    unknown_labels = set(aligned["label"]) - set(label_to_index)

    if unknown_labels:
        raise ValueError(f"Unknown labels: {sorted(unknown_labels)}")

    labels = aligned["label"].map(label_to_index).to_numpy(dtype=np.int64)

    representations["metadata"] = metadata

    representations["prism2_metadata"] = np.column_stack(
        [
            representations["prism2"],
            metadata,
        ]
    ).astype(np.float32)

    grouping_ids = aligned["grouping_id"].astype(str).to_numpy()

    split_names = aligned["split"].astype(str).to_numpy()

    for split_name, expected_count in {
        "train": 141,
        "validation": 21,
        "test": 42,
    }.items():
        observed_count = int((split_names == split_name).sum())

        if observed_count != expected_count:
            raise ValueError(
                f"{split_name}: expected {expected_count}, found {observed_count}"
            )

    grouping_table = pd.DataFrame(
        {
            "grouping_id": grouping_ids,
            "split": split_names,
        }
    )

    leakage = grouping_table.groupby("grouping_id")["split"].nunique()

    if (leakage > 1).any():
        raise ValueError("A grouping ID appears in multiple splits.")

    return BaselineDataset(
        patient_ids=reference_patient_ids,
        grouping_ids=grouping_ids,
        wsi_ids=reference_wsi_ids,
        labels=labels,
        label_names=aligned["label"].astype(str).to_numpy(),
        splits=split_names,
        representations=representations,
    )
