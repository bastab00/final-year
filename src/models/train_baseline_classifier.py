{"id":"92411","variant":"standard","title":"train_baseline_classifier.py — Baseline TF-IDF Model"}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# ---------------------------------------------------------
# 1. Load cleaned dataset
# ---------------------------------------------------------
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")

X = df["Cleaned_Abstract"]
y = df["Label"]

# ---------------------------------------------------------
# 2. Train-test split (stratified because labels are imbalanced)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# ---------------------------------------------------------
# 3. TF-IDF Vectorizer
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# ---------------------------------------------------------
# 4. Logistic Regression (Baseline Classifier)
# ---------------------------------------------------------
model = LogisticRegression(max_iter=500, class_weight="balanced")

model.fit(X_train_tfidf, y_train)

# ---------------------------------------------------------
# 5. Predictions & Evaluation
# ---------------------------------------------------------
y_pred = model.predict(X_test_tfidf)

print("\n===== BASELINE CLASSIFIER REPORT =====\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# ---------------------------------------------------------
# 6. Save model + vectorizer
# ---------------------------------------------------------
pickle.dump(model, open("../../artifacts/models/baseline_lr_model.pkl", "wb"))
pickle.dump(vectorizer, open("../../artifacts/models/tfidf_vectorizer.pkl", "wb"))

print("\n[INFO] Saved baseline_lr_model.pkl and tfidf_vectorizer.pkl")
