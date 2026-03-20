import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (with Year column already present)
df = pd.read_csv("../../data/processed/dataset_with_years.csv")

# Drop missing years
df = df.dropna(subset=["Year"])

# Ensure Year is integer
df["Year"] = df["Year"].astype(int)

# Count papers per year
pub_trend = df.groupby("Year").size().reset_index(name="Number_of_Papers")

# Sort by year
pub_trend = pub_trend.sort_values("Year")

# Plot
plt.figure(figsize=(10,5))
plt.plot(pub_trend["Year"], pub_trend["Number_of_Papers"], marker="o")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.title("Publication Trends Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
