from pathlib import Path

import pandas as pd

MANIFEST_PATH = Path("artifacts/manifests/patient_manifest.csv")
EXCLUSIONS_PATH = Path("artifacts/manifests/excluded_patients.csv")
INVENTORY_PATH = Path("artifacts/manifests/dataset_image_inventory.csv")
SPLITS_PATH = Path("artifacts/splits/patient_splits.csv")


def test_manifest_and_exclusions_cover_210_patients() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    exclusions = pd.read_csv(EXCLUSIONS_PATH)

    assert len(manifest) == 204
    assert len(exclusions) == 6
    assert len(manifest) + len(exclusions) == 210

    included_ids = set(manifest["patient_id"])
    excluded_ids = set(exclusions["patient_id"])

    assert included_ids.isdisjoint(excluded_ids)


def test_every_selected_image_exists_in_inventory() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)

    selected_ids = set(manifest["selected_wsi_id"])
    available_ids = set(inventory["wsi_id"])

    assert selected_ids.issubset(available_ids)
    assert manifest["selected_wsi_id"].is_unique


def test_split_contains_every_included_patient_once() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    splits = pd.read_csv(SPLITS_PATH)

    assert len(splits) == len(manifest)
    assert splits["patient_id"].is_unique
    assert set(splits["patient_id"]) == set(manifest["patient_id"])
    assert set(splits["split"]) == {
        "train",
        "validation",
        "test",
    }


def test_grouping_ids_do_not_cross_splits() -> None:
    splits = pd.read_csv(
        SPLITS_PATH,
        dtype={"grouping_id": "string"},
    )

    splits_per_group = splits.groupby("grouping_id")["split"].nunique()

    assert splits_per_group.max() == 1


def test_all_classes_exist_in_every_split() -> None:
    splits = pd.read_csv(SPLITS_PATH)

    table = pd.crosstab(splits["label"], splits["split"])

    assert table.shape == (7, 3)
    assert (table > 0).all().all()


def test_shared_identifier_stays_together() -> None:
    splits = pd.read_csv(
        SPLITS_PATH,
        dtype={"grouping_id": "string"},
    )

    duplicated = splits.loc[splits["grouping_id"] == "8377886"]

    assert len(duplicated) == 2
    assert duplicated["split"].nunique() == 1
