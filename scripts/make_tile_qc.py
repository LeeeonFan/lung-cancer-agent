from pathlib import Path

import pandas as pd
import pyvips
from PIL import Image, ImageDraw

TILE_MANIFEST_PATH = Path("artifacts/manifests/tile_manifest.csv.gz")
SPLITS_PATH = Path("artifacts/splits/patient_splits.csv")
DOWNSAMPLE_PATH = Path("artifacts/manifests/downsample_manifest.csv")

SELECTION_PATH = Path("artifacts/manifests/tile_qc_selection.csv")
OUTPUT_DIRECTORY = Path("data/processed/qc_tiles")
CONTACT_SHEET_PATH = OUTPUT_DIRECTORY / "contact_sheet.jpg"

TARGET_FRACTIONS = {
    "boundary": 0.52,
    "medium": 0.75,
    "high": 0.95,
}
TILES_PER_GROUP = 8

TILE_DISPLAY_SIZE = 224
CARD_WIDTH = 250
CARD_HEIGHT = 290
COLUMNS = 4


def select_qc_tiles(
    train_tiles: pd.DataFrame,
) -> pd.DataFrame:
    selections: list[pd.DataFrame] = []

    for group_name, target_fraction in TARGET_FRACTIONS.items():
        candidates = train_tiles.copy()
        candidates["distance"] = (candidates["tissue_fraction"] - target_fraction).abs()

        selected = (
            candidates.sort_values(["distance", "wsi_id", "selection_rank"])
            .drop_duplicates("wsi_id")
            .head(TILES_PER_GROUP)
            .copy()
        )

        selected["qc_group"] = group_name
        selected["target_fraction"] = target_fraction
        selections.append(selected)

    result = pd.concat(
        selections,
        ignore_index=True,
    )

    assert len(result) == (len(TARGET_FRACTIONS) * TILES_PER_GROUP)

    return result


def extract_tiles(selection: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths: list[str] = []

    for index, row in selection.iterrows():
        source_path = Path(row["slide_path"])
        tile_size = int(row["tile_size"])
        x = int(row["x"])
        y = int(row["y"])

        output_name = f"{row['qc_group']}_{row['tile_id']}.jpg"
        output_path = OUTPUT_DIRECTORY / output_name

        print(f"[{index + 1}/{len(selection)}] {row['qc_group']}: {row['tile_id']}")

        slide = pyvips.Image.new_from_file(
            str(source_path),
            access="random",
        )
        tile = slide.crop(
            x,
            y,
            tile_size,
            tile_size,
        )
        tile.jpegsave(
            str(output_path),
            Q=95,
            optimize_coding=True,
        )

        output_paths.append(str(output_path))

    result = selection.copy()
    result["tile_path"] = output_paths
    return result


def create_contact_sheet(selection: pd.DataFrame) -> None:
    rows = (len(selection) + COLUMNS - 1) // COLUMNS

    sheet = Image.new(
        "RGB",
        (
            COLUMNS * CARD_WIDTH,
            rows * CARD_HEIGHT,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for position, row in selection.reset_index(drop=True).iterrows():
        column = position % COLUMNS
        grid_row = position // COLUMNS

        left = column * CARD_WIDTH
        top = grid_row * CARD_HEIGHT

        with Image.open(row["tile_path"]) as image:
            tile = image.convert("RGB")

        sheet.paste(
            tile,
            (left + 13, top + 8),
        )

        draw.text(
            (left + 13, top + TILE_DISPLAY_SIZE + 14),
            (f"{row['qc_group']} | {row['tissue_fraction']:.2f}"),
            fill="black",
        )
        draw.text(
            (left + 13, top + TILE_DISPLAY_SIZE + 34),
            f"{row['wsi_id']} | {row['label']}",
            fill="black",
        )

    sheet.save(
        CONTACT_SHEET_PATH,
        quality=92,
    )


def main() -> None:
    tiles = pd.read_csv(TILE_MANIFEST_PATH)
    splits = pd.read_csv(SPLITS_PATH)
    downsample = pd.read_csv(DOWNSAMPLE_PATH)

    slide_information = splits[
        [
            "patient_id",
            "selected_wsi_id",
            "label",
            "split",
        ]
    ].rename(columns={"selected_wsi_id": "wsi_id"})

    slide_paths = downsample[["wsi_id", "output_path"]].rename(
        columns={"output_path": "slide_path"}
    )

    tiles = tiles.drop(columns=["split"]).merge(
        slide_information,
        on=["patient_id", "wsi_id"],
        validate="many_to_one",
    )
    tiles = tiles.merge(
        slide_paths,
        on="wsi_id",
        validate="many_to_one",
    )

    train_tiles = tiles.loc[tiles["split"] == "train"].copy()

    assert not train_tiles.empty
    assert set(train_tiles["split"]) == {"train"}

    selection = select_qc_tiles(train_tiles)
    selection = extract_tiles(selection)

    selection.to_csv(
        SELECTION_PATH,
        index=False,
    )
    create_contact_sheet(selection)

    print()
    print("=== Tile QC complete ===")
    print(f"Selected tiles: {len(selection)}")
    print(
        selection.groupby("qc_group")["tissue_fraction"]
        .agg(["min", "mean", "max"])
        .to_string()
    )
    print()
    print(f"Contact sheet: {CONTACT_SHEET_PATH}")
    print(f"Selection: {SELECTION_PATH}")


if __name__ == "__main__":
    main()
