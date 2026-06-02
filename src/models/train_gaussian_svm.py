# train_gaussian_svm.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ---------------------------------------------------------
# 1. Load cleaned dataset
# ---------------------------------------------------------
df = pd.read_csv("../../data/processed/dataset_latest_version.csv")

X = df["Cleaned_Abstract"]
y = df["Label"]

# ---------------------------------------------------------
# 2. Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 3. TF-IDF Vectorizer
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------------
# 4. Gaussian (RBF) SVM
# ---------------------------------------------------------
model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# ---------------------------------------------------------
# 5. Predictions & Evaluation
# ---------------------------------------------------------
y_pred = model.predict(X_test_tfidf)

print("\n===== GAUSSIAN SVM REPORT =====\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# ---------------------------------------------------------
# 6. Save model + vectorizer
# ---------------------------------------------------------
joblib.dump(
    model,
    "../../artifacts/models/gaussian_svm_model_latest.pkl"
)

joblib.dump(
    vectorizer,
    "../../artifacts/models/gaussian_svm_tfidf_vectorizer_latest.pkl"
)

print("\n[INFO] Saved gaussian_svm_model_latest.pkl and gaussian_svm_tfidf_vectorizer_latest.pkl")