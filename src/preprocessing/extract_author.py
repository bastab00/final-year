import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load dataset
df = pd.read_csv("../../data/processed/dataset_with_years.csv")
dois = df["DOI"].dropna().unique().tolist()

def fetch_authors(doi):
    url = f"https://api.crossref.org/works/{doi}"
    records = []

    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            msg = r.json()["message"]
            for a in msg.get("author", []):
                given = a.get("given", "").strip()
                family = a.get("family", "").strip()
                orcid = a.get("ORCID")

                full_name = f"{given} {family}".strip()
                std_name = f"{family}, {given[:1]}" if given and family else None

                records.append({
                    "DOI": doi,
                    "Author_Full": full_name,
                    "Author_Standardized": std_name,
                    "ORCID": orcid
                })
    except:
        pass

    return records

results = []

BATCH_SIZE = 20      # safe for Crossref
MAX_WORKERS = 10     # parallel requests per batch

for i in range(0, len(dois), BATCH_SIZE):
    batch = dois[i:i + BATCH_SIZE]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_authors, d) for d in batch]
        for future in as_completed(futures):
            results.extend(future.result())

    print(f"Processed {min(i+BATCH_SIZE, len(dois))}/{len(dois)} DOIs")

authors_df = pd.DataFrame(results)
authors_df.to_csv("../../data/processed/authors_extracted.csv", index=False)
