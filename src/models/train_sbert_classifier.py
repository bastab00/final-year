import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pickle

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")

X = df["Abstract"]    # USE RAW ABSTRACT for SBERT
y = df["Label"]

# ---------------------------------------------------------
# 2. Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 3. Load SBERT model
# ---------------------------------------------------------
print("[INFO] Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# 4. Encode abstracts into embeddings (768-d vectors)
# ---------------------------------------------------------
print("[INFO] Embedding training samples...")
X_train_embeddings = model.encode(X_train.tolist(), batch_size=16, show_progress_bar=True)

print("[INFO] Embedding test samples...")
X_test_embeddings = model.encode(X_test.tolist(), batch_size=16, show_progress_bar=True)

# ---------------------------------------------------------
# 5. Train SVM on SBERT embeddings
# ---------------------------------------------------------
classifier = LinearSVC(class_weight="balanced")
classifier.fit(X_train_embeddings, y_train)

# ---------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------
y_pred = classifier.predict(X_test_embeddings)

print("\n===== SBERT + SVM CLASSIFIER REPORT =====\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# ---------------------------------------------------------
# 7. Save model + embeddings model
# ---------------------------------------------------------
pickle.dump(classifier, open("../../artifacts/models/sbert_svm_model.pkl", "wb"))
model.save("../../artifacts/models/sbert_embedding_model/")

print("\n[INFO] Saved sbert_svm_model.pkl and SBERT model folder.")
