from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/CLWD.csv")
TARGET_COLUMN = "Benchmark_Label_7class"

REQUIRED_COLUMNS = [
    "SampleNumber",
    "WSI_ID",
    "Sex",
    "Age",
    "WHO2021_category",
    TARGET_COLUMN,
]


def numeric_wsi_id(wsi_id: str) -> int:
    """Convert an ID such as WSI-346 to the integer 346."""
    return int(str(wsi_id).split("-")[-1])


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        dtype={
            "SampleNumber": "string",
            "WSI_ID": "string",
            "Sex": "string",
            TARGET_COLUMN: "string",
        },
    )

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("=== Basic structure ===")
    print(f"Rows/slides: {len(df)}")
    print(f"Unique WSI_ID values: {df['WSI_ID'].nunique()}")
    print(f"Unique SampleNumber values: {df['SampleNumber'].nunique()}")
    print()

    print("=== Missing values ===")
    print(df[REQUIRED_COLUMNS].isna().sum().to_string())
    print()

    print("=== Sex values ===")
    print(df["Sex"].value_counts(dropna=False).to_string())
    print()

    print("=== Age summary ===")
    print(df["Age"].describe().to_string())
    print()

    print("=== Slide-level target distribution ===")
    print(df[TARGET_COLUMN].value_counts(dropna=False).to_string())
    print()

    print("=== Slides per SampleNumber ===")
    print(df.groupby("SampleNumber").size().describe().to_string())
    print()

    duplicate_id = df.loc[
        df["SampleNumber"] == "8377886",
        [
            "SampleNumber",
            "WSI_ID",
            "Sex",
            "Age",
            "WHO2021_category",
            TARGET_COLUMN,
        ],
    ].sort_values("WSI_ID")

    print("=== Records for duplicated ID 8377886 ===")
    print(duplicate_id.to_string(index=False))
    print()

    # A SampleNumber is label-conflicting if its slides have
    # more than one distinct seven-class benchmark label.
    label_counts = df.groupby("SampleNumber")[TARGET_COLUMN].nunique(dropna=False)
    conflicting_ids = label_counts[label_counts > 1].index

    conflicts = df.loc[
        df["SampleNumber"].isin(conflicting_ids),
        ["SampleNumber", "WSI_ID", "Age", "Sex", TARGET_COLUMN],
    ].copy()

    conflicts["WSI_number"] = conflicts["WSI_ID"].map(numeric_wsi_id)
    conflicts = conflicts.sort_values(["SampleNumber", "WSI_number"])

    print("=== Conflicting patient labels ===")
    print(f"Number of conflicting SampleNumber values: {len(conflicting_ids)}")
    print(conflicts.drop(columns="WSI_number").to_string(index=False))
    print()

    # Count apparent patient records using demographics in addition
    # to SampleNumber. This is for auditing only, not yet the split key.
    record_columns = ["SampleNumber", "Age", "Sex"]
    apparent_records = df[record_columns].drop_duplicates()

    print("=== Apparent patient records ===")
    print(f"Unique combinations of SampleNumber, Age and Sex: {len(apparent_records)}")
    print()

    print("=== Unique labels ===")
    labels = sorted(df[TARGET_COLUMN].dropna().unique())
    print(labels)


if __name__ == "__main__":
    main()
