# train_calibrated_svm.py
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib

# ==============================
# Load and prepare dataset
# ==============================
print("[INFO] Loading dataset...")
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")
texts = df["Cleaned_Abstract"].astype(str).tolist()
labels = df["Label"].tolist()

# ==============================
# Vectorize texts
# ==============================
print("[INFO] Vectorizing abstracts...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
y = labels

# ==============================
# Train calibrated SVM
# ==============================
print("[INFO] Training calibrated LinearSVC...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svc = LinearSVC(class_weight='balanced')
calibrated_svc = CalibratedClassifierCV(svc)
calibrated_svc.fit(X_train, y_train)

# ==============================
# Save model and vectorizer
# ==============================
joblib.dump(calibrated_svc, "../../artifacts/models/svm_calibrated_model.pkl")
joblib.dump(vectorizer, "../../artifacts/models/svm_tfidf_vectorizer.pkl")
print("[INFO] Saved calibrated model and vectorizer.")
