import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap

# ================================
# Load cleaned dataset  
# ================================
print("[INFO] Loading data...")
df = pd.read_csv("../../../data/processed/cleaned_inflation_dataset.csv")

# Split by label
ml_docs = df[df["Label"] == 1]["Cleaned_Abstract"].astype(str).tolist()
non_ml_docs = df[df["Label"] == 0]["Cleaned_Abstract"].astype(str).tolist()

print(f"[INFO] ML papers: {len(ml_docs)}, Non-ML papers: {len(non_ml_docs)}")

# ================================
# Embedding model
# ================================
print("[INFO] Loading SBERT model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# UMAP config
umap_model = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42)

# ================================
# Define helper to train BERTopic with KMeans
# ================================
def train_topic_model(docs, n_clusters=10):
    embeddings = embedding_model.encode(docs, batch_size=32, show_progress_bar=True)
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=kmeans_model,
        verbose=True
    )

    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    return topic_model

# ================================
# Train on ML papers
# ================================
print("\n[INFO] Training topic model on ML papers...")
ml_topic_model = train_topic_model(ml_docs, n_clusters=10)

# ================================
# Train on non-ML papers
# ================================
print("\n[INFO] Training topic model on NON-ML papers...")
non_ml_topic_model = train_topic_model(non_ml_docs, n_clusters=10)

# ================================
# Display top 5 topics from each
# ================================
def print_top_topics(topic_model, label):
    print(f"\n===== Top 5 Topics — {label} =====")
    for i in range(5):
        topic_words = topic_model.get_topic(i)
        if topic_words:
            words = ", ".join([word for word, _ in topic_words])
            print(f"Topic {i}: {words}")

print_top_topics(ml_topic_model, "ML")
print_top_topics(non_ml_topic_model, "Non-ML")

# ================================
# Save models
# ================================
ml_topic_model.save("ml_topic_model")
non_ml_topic_model.save("nonml_topic_model")

print("\n[INFO] Contrastive topic modeling complete. Models saved.")
