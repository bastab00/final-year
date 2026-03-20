import pandas as pd
import matplotlib.pyplot as plt

# Load top authors
top_authors = pd.read_csv("../../data/processed/top_authors.csv")

# Take top 10
top10 = top_authors.head(10)

plt.figure(figsize=(10,5))
plt.barh(top10["Author_Standardized"], top10["Paper_Count"])
plt.xlabel("Number of Papers")
plt.ylabel("Author")
plt.title("Top 10 Most Productive Authors")
plt.gca().invert_yaxis()  # highest at top
plt.tight_layout()
plt.show()


plt.figure(figsize=(7,5))
plt.hist(top_authors["Paper_Count"], bins=20)
plt.xlabel("Number of Papers per Author")
plt.ylabel("Number of Authors")
plt.title("Author Productivity Distribution")
plt.tight_layout()
plt.show()
