import pandas as pd

# Load extracted authors file
authors_df = pd.read_csv("../../data/processed/authors_extracted.csv")

# Drop missing standardized names
authors_df = authors_df.dropna(subset=["Author_Standardized"])

# Count papers per author (unique DOI per author)
top_authors = (
    authors_df
    .drop_duplicates(subset=["DOI", "Author_Standardized"])
    .groupby("Author_Standardized")
    .size()
    .reset_index(name="Paper_Count")
    .sort_values("Paper_Count", ascending=False)
)

# Save results
top_authors.to_csv("../../data/processed/top_authors.csv", index=False)

# Show top 10
print(top_authors.head(10))
