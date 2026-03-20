import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap

# =====================================

# Load dataset

# =====================================

print("[INFO] Loading dataset...")
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")

# Extract documents

documents = df["Cleaned_Abstract"].astype(str).tolist()

# =====================================

# SBERT Embeddings

# =====================================

print("[INFO] Loading MiniLM model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("[INFO] Generating embeddings...")
embeddings = embedding_model.encode(
documents,
batch_size=32,
show_progress_bar=True
)

# =====================================

# Dimensionality Reduction (UMAP)

# =====================================

print("[INFO] Running UMAP...")
umap_model = umap.UMAP(
n_neighbors=15,
n_components=5,
metric="cosine",
random_state=42
)
reduced_embeddings = umap_model.fit_transform(embeddings)

# =====================================

# KMeans Clustering

# =====================================

NUM_TOPICS = 12
print(f"[INFO] Running KMeans clustering with {NUM_TOPICS} topics...")
kmeans_model = KMeans(n_clusters=NUM_TOPICS, random_state=42)
cluster_labels = kmeans_model.fit_predict(reduced_embeddings)

# =====================================

# BERTopic Model (No HDBSCAN)

# =====================================

print("[INFO] Training BERTopic without HDBSCAN...")
topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model, verbose=True)
topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)

# Overwrite BERTopic topics with KMeans clusters

topic_model.update_topics(documents, topics=cluster_labels)

# =====================================

# Display Topics

# =====================================

print("\n===== TOPIC SUMMARY =====\n")
topic_info = topic_model.get_topic_info()
print(topic_info)

NUM_TOP_KEYWORDS = 10
print("\n===== TOPICS AND TOP KEYWORDS =====\n")
for topic_num in topic_info['Topic']:
    if topic_num == -1:
        continue
    print(f"Topic {topic_num}:")
    topic_keywords = topic_model.get_topic(topic_num)
    top_words = ", ".join([word for word, _ in topic_keywords[:NUM_TOP_KEYWORDS]])
    print(f"Top Keywords: {top_words}\n")

# =====================================

# Save Model

# =====================================

topic_model.save("bertopic_kmeans_model")
print("\n[INFO] BERTopic model saved successfully!")
