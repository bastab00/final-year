import pandas as pd
from collections import Counter

# Load dataset
df = pd.read_csv("../../data/processed/dataset_with_years.csv")

# Drop missing abstracts
texts = df["Cleaned_Abstract"].dropna().tolist()

# Tokenize (already cleaned, space-separated)
tokens = []
for text in texts:
    tokens.extend(text.split())

# Count word frequencies
word_freq = Counter(tokens)

# Convert to DataFrame
keywords_df = (
    pd.DataFrame(word_freq.items(), columns=["Keyword", "Frequency"])
    .sort_values("Frequency", ascending=False)
)

# Save results
keywords_df.to_csv("../../data/processed/keyword_frequency.csv", index=False)

# Show top 20
print(keywords_df.head(20))


import matplotlib.pyplot as plt

top20 = keywords_df.head(20)

plt.figure(figsize=(10,5))
plt.barh(top20["Keyword"], top20["Frequency"])
plt.xlabel("Frequency")
plt.ylabel("Keyword")
plt.title("Top 20 Keywords in Abstracts")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
