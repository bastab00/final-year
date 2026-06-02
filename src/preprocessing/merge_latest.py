import os
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Latest dataset used by training
LATEST_DATASET = (
    ROOT_DIR /
    "data" /
    "processed" /
    "dataset_latest_version.csv"
)

# Fallback for first run
INITIAL_DATASET = (
    ROOT_DIR /
    "data" /
    "processed" /
    "dataset_v3.csv"
)

# Newly processed papers
NEW_DATASET = (
    ROOT_DIR /
    "data" /
    "processed" /
    "dataset_latest_version_without_merge.csv"
)

# Temporary file
TEMP_OUTPUT = (
    ROOT_DIR /
    "data" /
    "processed" /
    "dataset_latest_version_temp.csv"
)


def main():

    print("Loading datasets...")

    # Use latest dataset if available
    if LATEST_DATASET.exists():

        print(
            f"[INFO] Using existing latest dataset:"
            f"\n{LATEST_DATASET}"
        )

        original_df = pd.read_csv(
            LATEST_DATASET
        )

    else:

        print(
            f"[INFO] First run detected."
            f"\nUsing {INITIAL_DATASET}"
        )

        original_df = pd.read_csv(
            INITIAL_DATASET
        )

    new_df = pd.read_csv(
        NEW_DATASET
    )

    print(
        f"\nExisting Dataset : {len(original_df)}"
    )

    print(
        f"New Dataset      : {len(new_df)}"
    )

    required_cols = [
        "DOI",
        "Abstract",
        "Label",
        "Cleaned_Abstract"
    ]

    original_df = original_df[
        required_cols
    ]

    new_df = new_df[
        required_cols
    ]

    merged_df = pd.concat(
        [original_df, new_df],
        ignore_index=True
    )

    before_dedup = len(
        merged_df
    )

    merged_df = merged_df.drop_duplicates(
        subset=["DOI"],
        keep="first"
    )

    merged_df = merged_df.drop_duplicates(
        subset=["Cleaned_Abstract"],
        keep="first"
    )

    after_dedup = len(
        merged_df
    )

    # Save temporary file
    merged_df.to_csv(
        TEMP_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    # Validate temp file
    check_df = pd.read_csv(
        TEMP_OUTPUT
    )

    if len(check_df) < len(original_df):

        raise Exception(
            "Merged dataset is smaller than "
            "existing dataset. Aborting."
        )

    # Atomic replacement
    os.replace(
        TEMP_OUTPUT,
        LATEST_DATASET
    )

    print("\n========== SUMMARY ==========")

    print(
        f"Existing Rows      : {len(original_df)}"
    )

    print(
        f"New Rows           : {len(new_df)}"
    )

    print(
        f"Merged Rows        : {before_dedup}"
    )

    print(
        f"Final Rows         : {after_dedup}"
    )

    print(
        f"Duplicates Removed : "
        f"{before_dedup - after_dedup}"
    )

    print("\nLabel Distribution:")

    print(
        merged_df["Label"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nUpdated Dataset:"
        f"\n{LATEST_DATASET}"
    )

    print("=============================")


if __name__ == "__main__":
    main()