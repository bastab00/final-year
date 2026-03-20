# predict_all_models.py
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

print("[INFO] Loading models and vectorizers...")

# Model and vectorizer paths
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

# Input abstract
print("\nEnter abstract text (press Enter when done):\n")
user_input = input("> ")

# Run prediction with each model
print("\n📌 Prediction Results:")
for name, paths in models_info.items():
    try:
        model = joblib.load(paths["model"])
        vectorizer = joblib.load(paths["vectorizer"])
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0] if hasattr(model, "predict_proba") else [None, None]
        print(f"\n--- {name} ---")
        print(f"Predicted Label: {prediction}")
        if proba[0] is not None:
            print(f"Class Probabilities: [{proba[0]:.3f}, {proba[1]:.3f}]")
        else:
            print("Class Probabilities: Not available")
    except Exception as e:
        print(f"\n--- {name} ---")
        print(f"[ERROR] Failed to run prediction: {e}")
