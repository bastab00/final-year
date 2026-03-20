import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")
dois = df["DOI"].dropna().unique()

def fetch_year(doi):
    try:
        url = f"https://api.crossref.org/works/{doi}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            msg = r.json()["message"]
            for key in ["published-print", "published-online", "issued"]:
                if key in msg and "date-parts" in msg[key]:
                    return doi, msg[key]["date-parts"][0][0]
    except:
        pass
    return doi, None

results = {}

BATCH_SIZE = 20        # safe
MAX_WORKERS = 10       # fast but polite

for i in range(0, len(dois), BATCH_SIZE):
    batch = dois[i:i+BATCH_SIZE]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_year, d) for d in batch]
        for future in as_completed(futures):
            doi, year = future.result()
            results[doi] = year

    print(f"Processed {min(i+BATCH_SIZE, len(dois))}/{len(dois)} DOIs")

df["Year"] = df["DOI"].map(results)
df.to_csv("../../data/processed/dataset_with_years.csv", index=False)
