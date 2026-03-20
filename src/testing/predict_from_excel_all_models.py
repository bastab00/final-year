# predict_from_excel_all_models.py
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load Excel
print("[INFO] Reading unseen data from Excel...")
df = pd.read_excel("../../data/raw/unseen_labeled.xlsx")  # Adjust path as needed
assert "Abstract" in df.columns, "Missing 'Abstract' column in Excel."

# Setup models and vectorizers
print("[INFO] Loading models and vectorizers...")
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

# Load SBERT model
print("[INFO] Loading SBERT model...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
sbert_clf = joblib.load("../../artifacts/models/sbert_lr_classifier.pkl")

# Predict for each abstract
results = []

print("[INFO] Running predictions...")
for idx, row in df.iterrows():
    abstract = row["Abstract"]
    row_result = {"DOI": row["DOI"]}

    # Predict using traditional models
    for model_name, paths in models_info.items():
        try:
            model = joblib.load(paths["model"])
            vectorizer = joblib.load(paths["vectorizer"])
            vec = vectorizer.transform([abstract])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else [None, None]

            row_result[f"{model_name} Prediction"] = pred
            row_result[f"{model_name} Prob_0"] = proba[0] if proba[0] is not None else "N/A"
            row_result[f"{model_name} Prob_1"] = proba[1] if proba[1] is not None else "N/A"
        except Exception as e:
            row_result[f"{model_name} Prediction"] = f"Error: {e}"
            row_result[f"{model_name} Prob_0"] = "Error"
            row_result[f"{model_name} Prob_1"] = "Error"

    # Predict using SBERT
    try:
        emb = sbert_model.encode([abstract])
        pred_sbert = sbert_clf.predict(emb)[0]
        proba_sbert = sbert_clf.predict_proba(emb)[0]

        row_result["SBERT Prediction"] = pred_sbert
        row_result["SBERT Prob_0"] = proba_sbert[0]
        row_result["SBERT Prob_1"] = proba_sbert[1]
    except Exception as e:
        row_result["SBERT Prediction"] = f"Error: {e}"
        row_result["SBERT Prob_0"] = "Error"
        row_result["SBERT Prob_1"] = "Error"

    results.append(row_result)

# Save results to Excel
output_df = pd.DataFrame(results)
output_path = "../../artifacts/outputs/predicted_results_all_models.xlsx"
output_df.to_excel(output_path, index=False)
print(f"[INFO] Saved results to {output_path}")
