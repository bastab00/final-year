{"id":"51022","variant":"standard","title":"train_svm_classifier.py — Linear SVM with Balanced Class Weights"}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# ---------------------------------------------------------
# 1. Load cleaned dataset
# ---------------------------------------------------------
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")

X = df["Cleaned_Abstract"]
y = df["Label"]

# ---------------------------------------------------------
# 2. Stratified train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 3. TF-IDF Vectorizer (shared config with baseline)
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1,2),
    stop_words="english",
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# ---------------------------------------------------------
# 4. Train Linear SVM (balanced)
# ---------------------------------------------------------
model = LinearSVC(class_weight="balanced")

model.fit(X_train_tfidf, y_train)

# ---------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------
y_pred = model.predict(X_test_tfidf)

print("\n===== LINEAR SVM CLASSIFIER REPORT =====\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# ---------------------------------------------------------
# 6. Save Model
# ---------------------------------------------------------
pickle.dump(model, open("../../artifacts/models/svm_balanced_model.pkl", "wb"))
pickle.dump(vectorizer, open("../../artifacts/models/svm_tfidf_vectorizer.pkl", "wb"))

print("\n[INFO] Saved svm_balanced_model.pkl and svm_tfidf_vectorizer.pkl")
