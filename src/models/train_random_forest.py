import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# ============================
#  LOAD CLEANED DATASET
# ============================
DATA_PATH = "../../data/processed/cleaned_inflation_dataset.csv"

print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

X = df["Cleaned_Abstract"]
y = df["Label"]

# ============================
#  TRAIN/TEST SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.17, random_state=42, stratify=y
)

# ============================
#  TF-IDF VECTORIZATION
# ============================
print("[INFO] Generating TF-IDF vectors...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ============================
#  RANDOM FOREST CLASSIFIER
# ============================
print("[INFO] Training Random Forest Classifier...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train_tfidf, y_train)

# ============================
#  EVALUATION
# ============================
y_pred = rf_model.predict(X_test_tfidf)

print("\n===== RANDOM FOREST CLASSIFIER REPORT =====\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

# ============================
#  SAVE MODEL + VECTORIZER
# ============================
OUTPUT_MODEL = "../../artifacts/models/rf_model.pkl"
OUTPUT_VECT = "../../artifacts/models/rf_tfidf_vectorizer.pkl"

joblib.dump(rf_model, OUTPUT_MODEL)
joblib.dump(tfidf, OUTPUT_VECT)

print(f"\n[INFO] Saved {OUTPUT_MODEL} and {OUTPUT_VECT}")
