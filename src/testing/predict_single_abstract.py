import joblib
import numpy as np

# Load model and vectorizer (CHANGE THESE PATHS PER MODEL)
model_path = "../../artifacts/models/baseline_lr_model.pkl"
vectorizer_path = "../../artifacts/models/tfidf_vectorizer.pkl"

print("[INFO] Loading model and vectorizer...")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Input abstract manually
print("\nEnter abstract text (press Enter when done):\n")
abstract = input("> ")

# Transform and predict
X = vectorizer.transform([abstract])
prediction = model.predict(X)
proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

# Output prediction
print("\n📌 Prediction Result:")
print("Predicted Label:", prediction[0])

if proba is not None:
    print("Class Probabilities:", np.round(proba[0], 3))
