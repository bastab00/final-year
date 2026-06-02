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
    "inflation_dataset_500.csv"
)

TARGET_POSITIVE = 250
TARGET_NEGATIVE = 250

POSITIVE_QUERIES = [
    "inflation forecasting",
    "inflation prediction",
    "inflation expectations",
    "consumer price index",
    "inflation targeting"
]

NEGATIVE_QUERIES = [
    "stock market prediction",
    "corporate finance",
    "labour economics",
    "economic growth",
    "exchange rates"
]


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


def collect_group(
    queries,
    label,
    target_count,
    seen_dois
):

    collected = []

    per_query_target = (
        target_count // len(queries)
    )

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

                if len(abstract) < 100:
                    continue

                doi = paper.get("doi")

                if doi:
                    doi = doi.replace(
                        "https://doi.org/",
                        ""
                    )

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

            if len(collected) >= target_count:
                break

            if page > 20:
                break

    return collected


def main():

    seen_dois = set()

    print(
        "\nCollecting Positive Papers..."
    )

    positive = collect_group(
        POSITIVE_QUERIES,
        1,
        TARGET_POSITIVE,
        seen_dois
    )

    print(
        "\nCollecting Negative Papers..."
    )

    negative = collect_group(
        NEGATIVE_QUERIES,
        0,
        TARGET_NEGATIVE,
        seen_dois
    )

    all_rows = positive + negative

    df = pd.DataFrame(
        all_rows,
        columns=[
            "DOI",
            "Abstract",
            "Label",
            "Cleaned_Abstract"
        ]
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n======================")
    print(f"Total: {len(df)}")
    print(
        f"Positive: "
        f"{(df['Label']==1).sum()}"
    )
    print(
        f"Negative: "
        f"{(df['Label']==0).sum()}"
    )
    print(f"Saved: {OUTPUT_FILE}")
    print("======================")


if __name__ == "__main__":
    main()