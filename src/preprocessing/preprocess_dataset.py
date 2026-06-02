import pandas as pd
import re
import string
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# INPUT = newly collected 500-paper dataset
INPUT_FILE = ROOT_DIR / "data" / "raw" / "inflation_dataset_latest_version.csv"

# OUTPUT = processed version
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "dataset_latest_version_without_merge.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():

    print(f"Loading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    initial_count = len(df)

    # Remove null abstracts
    df = df.dropna(subset=["Abstract"])

    # Remove empty abstracts
    df = df[
        df["Abstract"]
        .astype(str)
        .str.strip()
        .ne("")
    ]

    # Remove duplicate DOI
    if "DOI" in df.columns:
        df = df.drop_duplicates(
            subset=["DOI"],
            keep="first"
        )

    # Remove duplicate abstracts
    df = df.drop_duplicates(
        subset=["Abstract"],
        keep="first"
    )

    # Regenerate cleaned abstracts
    df["Cleaned_Abstract"] = (
        df["Abstract"]
        .astype(str)
        .apply(clean_text)
    )

    # Remove rows where cleaning produced nothing
    df = df[
        df["Cleaned_Abstract"]
        .str.len() > 20
    ]

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n========== SUMMARY ==========")
    print(f"Original Rows : {initial_count}")
    print(f"Final Rows    : {len(df)}")
    print(f"Positive      : {(df['Label']==1).sum()}")
    print(f"Negative      : {(df['Label']==0).sum()}")
    print(f"Saved To      : {OUTPUT_FILE}")
    print("=============================")


if __name__ == "__main__":
    main()