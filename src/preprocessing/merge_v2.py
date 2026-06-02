import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Original dataset
ORIGINAL_DATASET = (
    ROOT_DIR /
    "data" /
    "processed" /
    "cleaned_inflation_dataset.csv"
)

# Newly collected and processed papers
NEW_DATASET = (
    ROOT_DIR /
    "data" /
    "processed" /
    "processed_dataset_500.csv"
)

# Final merged dataset
OUTPUT_DATASET = (
    ROOT_DIR /
    "data" /
    "processed" /
    "dataset_v2.csv"
)


def main():

    print("Loading datasets...")

    original_df = pd.read_csv(ORIGINAL_DATASET)
    new_df = pd.read_csv(NEW_DATASET)

    print(f"Original Dataset : {len(original_df)}")
    print(f"New Dataset      : {len(new_df)}")

    # Ensure same columns
    required_cols = [
        "DOI",
        "Abstract",
        "Label",
        "Cleaned_Abstract"
    ]

    original_df = original_df[required_cols]
    new_df = new_df[required_cols]

    # Merge
    merged_df = pd.concat(
        [original_df, new_df],
        ignore_index=True
    )

    before_dedup = len(merged_df)

    # Remove duplicate DOI
    merged_df = merged_df.drop_duplicates(
        subset=["DOI"],
        keep="first"
    )

    # Remove duplicate cleaned abstracts
    merged_df = merged_df.drop_duplicates(
        subset=["Cleaned_Abstract"],
        keep="first"
    )

    after_dedup = len(merged_df)

    merged_df.to_csv(
        OUTPUT_DATASET,
        index=False,
        encoding="utf-8"
    )

    print("\n========== SUMMARY ==========")
    print(f"Original Rows : {len(original_df)}")
    print(f"New Rows      : {len(new_df)}")
    print(f"Merged Rows   : {before_dedup}")
    print(f"Final Rows    : {after_dedup}")
    print(f"Duplicates Removed : {before_dedup - after_dedup}")

    if "Label" in merged_df.columns:
        print("\nLabel Distribution:")
        print(merged_df["Label"].value_counts())

    print(f"\nSaved To : {OUTPUT_DATASET}")
    print("=============================")


if __name__ == "__main__":
    main()