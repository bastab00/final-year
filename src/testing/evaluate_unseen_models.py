# evaluate_unseen_models.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================
# Load unseen labeled data
# ==============================
df = pd.read_excel("../../data/raw/unseen_labeled.xlsx")  # Update path if needed
texts = df["Abstract"].astype(str).tolist()
labels = df["Label"].tolist()

# ==============================
# Define model paths and names
# ==============================
models_info = {
    "Logistic Regression": {
        "model": "../../artifacts/models/baseline_lr_model.pkl",
        "vectorizer": "../../artifacts/models/tfidf_vectorizer.pkl"
    },
    "Random Forest": {
        "model": "../../artifacts/models/rf_model.pkl",
        "vectorizer": "../../artifacts/models/rf_tfidf_vectorizer.pkl"
    },
    "SVM (Calibrated)": {
        "model": "../../artifacts/models/svm_calibrated_model.pkl",
        "vectorizer": "../../artifacts/models/svm_tfidf_vectorizer.pkl"
    },
    "XGBoost": {
        "model": "../../artifacts/models/xgb_model.pkl",
        "vectorizer": "../../artifacts/models/xgb_tfidf_vectorizer.pkl"
    }
}

# ==============================
# Evaluate each model
# ==============================
for name, paths in models_info.items():
    print(f"\n--- {name} ---")
    model = joblib.load(paths["model"])
    vectorizer = joblib.load(paths["vectorizer"])
    X_unseen = vectorizer.transform(texts)

    predictions = model.predict(X_unseen)

    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1 Score:  {f1:.2f}")
