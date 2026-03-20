import pandas as pd
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ==============================
# Load test data
# ==============================
data = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")
texts = data["Cleaned_Abstract"].astype(str)
labels = data["Label"]

# ==============================
# Load vectorizer and model
# ==============================
vectorizer = joblib.load("../../artifacts/models/rf_tfidf_vectorizer.pkl")
model = joblib.load("../../artifacts/models/rf_model.pkl")

# ==============================
# Vectorize and predict
# ==============================
X_test = vectorizer.transform(texts)
y_test = labels
y_proba = model.predict_proba(X_test)[:, 1]

# ==============================
# Compute ROC and AUC
# ==============================
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# ==============================
# Plot
# ==============================
plt.figure()
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("../../artifacts/outputs/rf_roc_curve.png")
plt.show()
