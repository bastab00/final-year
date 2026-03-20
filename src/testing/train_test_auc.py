import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")
X_text = df["Cleaned_Abstract"]
y = df["Label"]

# Load vectorizer and transform text
vectorizer = joblib.load("../../artifacts/models/rf_tfidf_vectorizer.pkl")
X = vectorizer.transform(X_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load trained Random Forest model
rf_model = joblib.load("../../artifacts/models/rf_model.pkl")

# Predict probabilities
y_train_proba = rf_model.predict_proba(X_train)[:, 1]
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Compute AUC
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")

# Plot ROC curves
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {train_auc:.2f})")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {test_auc:.2f})", linestyle="--")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve - Random Forest (Train vs Test)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
