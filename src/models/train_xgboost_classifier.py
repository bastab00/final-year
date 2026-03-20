import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

print("[INFO] Loading cleaned dataset...")
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")

X = df["Cleaned_Abstract"]
y = df["Label"]

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.17, random_state=42, stratify=y
)

# -------------------------
# TF-IDF Vectorizer
# -------------------------
print("[INFO] Creating TF-IDF matrix...")
tfidf = TfidfVectorizer(max_features=6000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -------------------------
# XGBoost Classifier
# -------------------------
print("[INFO] Training XGBoost classifier...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    scale_pos_weight=1.5,   # helps minority class
    n_jobs=-1,
    random_state=42
)

xgb_model.fit(X_train_tfidf, y_train)

# -------------------------
# Evaluation
# -------------------------
y_pred = xgb_model.predict(X_test_tfidf)

print("\n===== XGBOOST CLASSIFIER REPORT =====\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

# -------------------------
# Save
# -------------------------
joblib.dump(xgb_model, "../../artifacts/models/xgb_model.pkl")
joblib.dump(tfidf, "../../artifacts/models/xgb_tfidf_vectorizer.pkl")

print("\n[INFO] Saved xgb_model.pkl and xgb_tfidf_vectorizer.pkl")
