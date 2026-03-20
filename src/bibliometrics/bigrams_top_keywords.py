import pandas as pd
from collections import Counter

# Load dataset
df = pd.read_csv("../../data/processed/dataset_with_years.csv")

texts = df["Cleaned_Abstract"].dropna().tolist()

bigrams = []

for text in texts:
    words = text.split()
    bigrams.extend(zip(words, words[1:]))

# Join words with underscore
bigrams = ["_".join(bg) for bg in bigrams]

# Count frequency
bigram_freq = Counter(bigrams)

# Convert to DataFrame
bigrams_df = (
    pd.DataFrame(bigram_freq.items(), columns=["Bigram", "Frequency"])
    .sort_values("Frequency", ascending=False)
)

# Save
bigrams_df.to_csv("../../data/processed/bigram_frequency.csv", index=False)

# Show top 20
print(bigrams_df.head(20))

import matplotlib.pyplot as plt

top20 = bigrams_df.head(20)

plt.figure(figsize=(10,5))
plt.barh(top20["Bigram"], top20["Frequency"])
plt.xlabel("Frequency")
plt.ylabel("Bigram")
plt.title("Top 20 Bigrams in Abstracts")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
