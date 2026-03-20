import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split

# Load dataset
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

# Load Random Forest model
rf_model = joblib.load("../../artifacts/models/rf_model.pkl")

# ==============================
# CONFUSION MATRIX & REPORT
# ==============================

# Predict class labels
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Print confusion matrices
print("=== Train Confusion Matrix ===")
print(confusion_matrix(y_train, y_train_pred))

print("\n=== Test Confusion Matrix ===")
print(confusion_matrix(y_test, y_test_pred))

# Print classification reports
print("\n=== Train Classification Report ===")
print(classification_report(y_train, y_train_pred))

print("\n=== Test Classification Report ===")
print(classification_report(y_test, y_test_pred))

# ==============================
# PRECISION-RECALL CURVE
# ==============================

# Predict probabilities for test set
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Compute Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
avg_precision = average_precision_score(y_test, y_test_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"Average Precision = {avg_precision:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Random Forest")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
