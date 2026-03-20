from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

# Load labeled training data
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")
texts = df["Cleaned_Abstract"].astype(str).tolist()
labels = df["Label"]

# Encode using SBERT
print("[INFO] Encoding with SBERT...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sbert.encode(texts)

# Train classifier
print("[INFO] Training logistic regression classifier on SBERT embeddings...")
clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(embeddings, labels)

# Save model
joblib.dump(clf, "../../artifacts/models/sbert_lr_classifier.pkl")
print("[INFO] Saved SBERT classifier (LogisticRegression).")
