# shap_explain.py
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Load trained model and original vectorizer
# ==============================
print("[INFO] Loading model and vectorizer...")
model = joblib.load("../../artifacts/models/svm_calibrated_model.pkl")  # trained with 5000 features
vectorizer = joblib.load("../../artifacts/models/svm_tfidf_vectorizer.pkl")  # trained vectorizer

# ==============================
# Load dataset
# ==============================
print("[INFO] Loading dataset...")
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")
texts = df["Cleaned_Abstract"].astype(str).tolist()

# ==============================
# Vectorize using loaded vectorizer
# ==============================
print("[INFO] Vectorizing text data...")
X = vectorizer.transform(texts[:50])  # keep it small for memory
X_dense = X.toarray()

# ==============================
# Setup SHAP KernelExplainer
# ==============================
print("[INFO] Running SHAP explainability...")
background = X_dense[:10]  # small background sample
explainer = shap.KernelExplainer(model.predict_proba, background)

# Explain a single instance
instance_to_explain = X_dense[0:1]
shap_values = explainer(instance_to_explain)

# ==============================
# Save and show SHAP explanation
# ==============================
print("[INFO] Saving SHAP explanation...")

from shap import Explanation

# Prepare data for one instance and one class (class 1)
instance_index = 0
class_index = 1  # for binary classification: 0 or 1
feature_names = vectorizer.get_feature_names_out()
shap_expl = Explanation(
    values=shap_values[0][:, class_index],
    base_values=explainer.expected_value[class_index],
    data=instance_to_explain[0],
    feature_names=feature_names
)

viz = shap.plots.force(shap_expl)
shap.save_html("../../artifacts/outputs/svm_shap_explanation.html", viz)
print("[INFO] SHAP explanation saved as svm_shap_explanation.html")

