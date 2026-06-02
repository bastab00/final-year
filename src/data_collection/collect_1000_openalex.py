import requests
import pandas as pd
import re
import string
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    ROOT_DIR /
    "data" /
    "raw" /
    "inflation_dataset_latest_version.csv"
)

EXISTING_DATASET = (
    ROOT_DIR /
    "data" /
    "processed" /
    "dataset_latest_version.csv"
)

TARGET_POSITIVE = 600
TARGET_NEGATIVE = 400

POSITIVE_QUERIES = [
    "inflation forecasting",
    "inflation prediction",
    "inflation expectations",
    "consumer price index",
    "inflation targeting",
    "core inflation",
    "headline inflation",
    "monetary policy inflation",
    "price stability",
    "inflation dynamics",
    "inflation persistence",
    "inflation uncertainty",
    "inflation modeling",
    "inflation nowcasting",
    "inflation forecast",
    "CPI forecasting",
    "consumer prices",
    "price inflation",
    "monetary policy",
    "inflation economics",
    "inflation measurement"
]

NEGATIVE_QUERIES = [
    "stock market prediction",
    "corporate finance",
    "labour economics",
    "economic growth",
    "exchange rates",
    "financial markets",
    "banking sector",
    "portfolio optimization",
    "asset pricing",
    "corporate governance",
    "cryptocurrency",
    "machine learning",
    "computer vision",
    "healthcare analytics",
    "supply chain management"
]


def load_existing_dois():

    try:
        df = pd.read_csv(EXISTING_DATASET)

        dois = set(
            df["DOI"]
            .dropna()
            .astype(str)
            .tolist()
        )

        print(
            f"[INFO] Loaded "
            f"{len(dois)} existing DOIs"
        )

        return dois

    except Exception as e:

        print(
            f"[WARNING] Could not load existing dataset: {e}"
        )

        return set()


def reconstruct_abstract(inverted_index):

    if not inverted_index:
        return ""

    words = {}

    for word, positions in inverted_index.items():

        for pos in positions:
            words[pos] = word

    return " ".join(
        words[i]
        for i in sorted(words)
    )


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def collect_papers(
    queries,
    label,
    target_count,
    seen_dois
):

    collected = []

    for query in queries:

        print(f"\nQuery: {query}")

        page = 1

        while len(collected) < target_count:

            response = requests.get(
                "https://api.openalex.org/works",
                params={
                    "search": query,
                    "per-page": 200,
                    "page": page
                },
                timeout=30
            )

            response.raise_for_status()

            results = response.json().get(
                "results",
                []
            )

            if not results:
                break

            added = 0

            for paper in results:

                abstract = reconstruct_abstract(
                    paper.get(
                        "abstract_inverted_index"
                    )
                )

                # Must have abstract
                if not abstract:
                    continue

                # Minimum abstract length
                if len(abstract.split()) < 50:
                    continue

                doi = paper.get("doi")

                # Must have DOI
                if not doi:
                    continue

                doi = doi.replace(
                    "https://doi.org/",
                    ""
                )

                # Must have title
                title = paper.get("title", "")

                if len(title.strip()) < 5:
                    continue

                if doi and doi in seen_dois:
                    continue

                if doi:
                    seen_dois.add(doi)

                collected.append({
                    "DOI": doi,
                    "Abstract": abstract,
                    "Label": label,
                    "Cleaned_Abstract":
                        clean_text(abstract)
                })

                added += 1

                if len(collected) >= target_count:
                    break

            print(
                f"Page {page} | "
                f"+{added} | "
                f"Total={len(collected)}"
            )

            page += 1

            time.sleep(0.5)

            if page > 50:
                break

    return collected


def main():

    seen_dois = load_existing_dois()

    print("\nCollecting Positive Papers...")

    positive = collect_papers(
        POSITIVE_QUERIES,
        1,
        TARGET_POSITIVE,
        seen_dois
    )

    print("\nCollecting Negative Papers...")

    negative = collect_papers(
        NEGATIVE_QUERIES,
        0,
        TARGET_NEGATIVE,
        seen_dois
    )

    df = pd.DataFrame(
        positive + negative,
        columns=[
            "DOI",
            "Abstract",
            "Label",
            "Cleaned_Abstract"
        ]
    )


    df = df.drop_duplicates(
    subset=["DOI"],
    keep="first"
    )

    df = df.drop_duplicates(
        subset=["Cleaned_Abstract"],
        keep="first"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n======================")
    print(f"Total Papers : {len(df)}")
    print("\nLabel Distribution:")
    print(df["Label"].value_counts())
    print(f"Saved To     : {OUTPUT_FILE}")
    print("======================")

if __name__ == "__main__":
    main()