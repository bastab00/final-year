import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
import re

print("[INFO] Loading data...")
df = pd.read_csv("../../../data/processed/cleaned_inflation_dataset.csv")

# Extract year from DOI using regex
df["Year"] = df["DOI"].str.extract(r"(\d{4})").astype(float)
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

print(f"[INFO] Dataset size after year extraction: {len(df)}")

# Split abstracts
ml_df = df[df["Label"] == 1].copy()
non_ml_df = df[df["Label"] == 0].copy()

print(f"[INFO] ML papers: {len(ml_df)}, Non-ML papers: {len(non_ml_df)}")

# Load models
print("[INFO] Loading topic models...")
ml_model = BERTopic.load("ml_topic_model")
nonml_model = BERTopic.load("nonml_topic_model")

# ========================
# Topics over time (ML)
# ========================
print("[INFO] Calculating topics over time for ML...")
ml_docs = ml_df["Cleaned_Abstract"].astype(str).tolist()
ml_timestamps = ml_df["Year"].tolist()
ml_topics_over_time = ml_model.topics_over_time(ml_docs, ml_timestamps, nr_bins=8)
ml_fig = ml_model.visualize_topics_over_time(ml_topics_over_time)
ml_fig.write_html("../../../artifacts/outputs/ml_topics_over_time.html")

# ========================
# Topics over time (Non-ML)
# ========================
print("[INFO] Calculating topics over time for Non-ML...")
nonml_docs = non_ml_df["Cleaned_Abstract"].astype(str).tolist()
nonml_timestamps = non_ml_df["Year"].tolist()
nonml_topics_over_time = nonml_model.topics_over_time(nonml_docs, nonml_timestamps, nr_bins=8)
nonml_fig = nonml_model.visualize_topics_over_time(nonml_topics_over_time)
nonml_fig.write_html("../../../artifacts/outputs/nonml_topics_over_time.html")

print("[INFO] Saved: ml_topics_over_time.html, nonml_topics_over_time.html")
