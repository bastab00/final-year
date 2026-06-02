from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd

# Load labeled training data
df = pd.read_csv("../../data/processed/dataset_latest_version.csv")
texts = df["Cleaned_Abstract"].astype(str).tolist()
labels = df["Label"]

# Encode using SBERT
print("[INFO] Encoding with SBERT...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sbert.encode(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Train classifier
print("[INFO] Training Logistic Regression classifier on SBERT embeddings...")
clf = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("\n" + "="*60)
print("SBERT + Logistic Regression Results")
print("="*60)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "../../artifacts/models/sbert_lr_classifier_latest.pkl")
print("\n[INFO] Saved SBERT classifier (LogisticRegression).")