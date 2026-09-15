from pathlib import Path

import pandas as pd
import pyvips
from PIL import Image, ImageDraw, ImageOps

SPLITS_PATH = Path("artifacts/splits/patient_splits.csv")
DOWNSAMPLE_MANIFEST_PATH = Path("artifacts/manifests/downsample_manifest.csv")
SELECTION_PATH = Path("artifacts/manifests/qc_train_selection.csv")

OUTPUT_DIRECTORY = Path("data/processed/qc_train/thumbnails")
CONTACT_SHEET_PATH = Path("data/processed/qc_train/contact_sheet.jpg")

THUMBNAIL_WIDTH = 1024
THUMBNAIL_HEIGHT = 768
SAMPLES_PER_CLASS = 2

CARD_WIDTH = 520
CARD_HEIGHT = 430
IMAGE_BOX_WIDTH = 480
IMAGE_BOX_HEIGHT = 350
COLUMNS = 2


def safe_label(label: str) -> str:
    return label.lower().replace(" ", "_").replace("/", "_")


def create_thumbnail(
    source_path: Path,
    output_path: Path,
) -> None:
    image = pyvips.Image.new_from_file(
        str(source_path),
        access="sequential",
    )

    scale = min(
        THUMBNAIL_WIDTH / image.width,
        THUMBNAIL_HEIGHT / image.height,
    )

    thumbnail = image.resize(
        scale,
        kernel="lanczos3",
    )

    thumbnail.jpegsave(
        str(output_path),
        Q=90,
        optimize_coding=True,
    )


def create_contact_sheet(selection: pd.DataFrame) -> None:
    rows = (len(selection) + COLUMNS - 1) // COLUMNS

    sheet = Image.new(
        "RGB",
        (
            COLUMNS * CARD_WIDTH,
            rows * CARD_HEIGHT,
        ),
        color="white",
    )

    draw = ImageDraw.Draw(sheet)

    for position, row in selection.reset_index(drop=True).iterrows():
        column = position % COLUMNS
        grid_row = position // COLUMNS

        left = column * CARD_WIDTH
        top = grid_row * CARD_HEIGHT

        thumbnail_path = Path(row["thumbnail_path"])

        with Image.open(thumbnail_path) as image:
            image = image.convert("RGB")
            fitted = ImageOps.contain(
                image,
                (IMAGE_BOX_WIDTH, IMAGE_BOX_HEIGHT),
                method=Image.Resampling.LANCZOS,
            )

        image_left = left + (CARD_WIDTH - fitted.width) // 2
        image_top = top + 10

        sheet.paste(
            fitted,
            (image_left, image_top),
        )

        text_top = top + IMAGE_BOX_HEIGHT + 20

        draw.text(
            (left + 20, text_top),
            f"{row['label']} — {row['wsi_id']}",
            fill="black",
        )
        draw.text(
            (left + 20, text_top + 22),
            f"Patient: {row['patient_id']}",
            fill="black",
        )

    CONTACT_SHEET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sheet.save(
        CONTACT_SHEET_PATH,
        quality=92,
    )


def main() -> None:
    splits = pd.read_csv(SPLITS_PATH)
    downsample_manifest = pd.read_csv(DOWNSAMPLE_MANIFEST_PATH)

    train = splits.loc[splits["split"] == "train"].copy()

    train = train.sort_values(["label", "selected_wsi_number"])

    selection = (
        train.groupby("label", group_keys=False)
        .head(SAMPLES_PER_CLASS)
        .reset_index(drop=True)
    )

    assert len(selection) == 14
    assert selection["label"].nunique() == 7
    assert set(selection["split"]) == {"train"}

    output_paths = dict(
        zip(
            downsample_manifest["wsi_id"],
            downsample_manifest["output_path"],
            strict=True,
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    thumbnail_paths: list[str] = []

    for index, row in selection.iterrows():
        wsi_id = row["selected_wsi_id"]
        label = row["label"]

        source_path = Path(output_paths[wsi_id])
        filename = f"{safe_label(label)}_{wsi_id}.jpg"
        thumbnail_path = OUTPUT_DIRECTORY / filename

        print(f"[{index + 1}/{len(selection)}] {label}: {wsi_id}")

        create_thumbnail(
            source_path,
            thumbnail_path,
        )
        thumbnail_paths.append(str(thumbnail_path))

    selection = selection.rename(columns={"selected_wsi_id": "wsi_id"})
    selection["thumbnail_path"] = thumbnail_paths

    SELECTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    selection[
        [
            "patient_id",
            "grouping_id",
            "wsi_id",
            "label",
            "split",
            "thumbnail_path",
        ]
    ].to_csv(SELECTION_PATH, index=False)

    create_contact_sheet(selection)

    print()
    print("=== QC thumbnails created ===")
    print(f"Slides: {len(selection)}")
    print(f"Classes: {selection['label'].nunique()}")
    print(f"Selection: {SELECTION_PATH}")
    print(f"Contact sheet: {CONTACT_SHEET_PATH}")


if __name__ == "__main__":
    main()
