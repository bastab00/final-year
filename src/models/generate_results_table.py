import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# ============================
# Load dataset
# ============================
df = pd.read_csv("../../data/processed/dataset_latest_version.csv")

X = df["Cleaned_Abstract"]
y = df["Label"]

# ============================
# Same split used during training
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

results = []

# ============================
# Evaluation Function
# ============================
def evaluate(model_name, model_path, vect_path=None, sbert=False):
    print(f"[INFO] Evaluating {model_name}...")

    if sbert:
        from sentence_transformers import SentenceTransformer

        model = joblib.load(model_path)
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        X_test_vec = encoder.encode(
            list(X_test),
            show_progress_bar=False
        )
    else:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vect_path)
        X_test_vec = vectorizer.transform(X_test)

    y_pred = model.predict(X_test_vec)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted"
    )

    accuracy = accuracy_score(y_test, y_pred)

    results.append([
        model_name,
        accuracy,
        precision,
        recall,
        f1
    ])

# ============================
# Evaluate Models
# ============================
evaluate(
    "Logistic Regression",
    "../../artifacts/models/baseline_lr_model_latest.pkl",
    "../../artifacts/models/tfidf_vectorizer_latest.pkl"
)

evaluate(
    "Gaussian SVM",
    "../../artifacts/models/gaussian_svm_model_latest.pkl",
    "../../artifacts/models/gaussian_svm_tfidf_vectorizer_latest.pkl"
)

evaluate(
    "Random Forest",
    "../../artifacts/models/rf_model_latest.pkl",
    "../../artifacts/models/rf_tfidf_vectorizer_latest.pkl"
)

evaluate(
    "XGBoost",
    "../../artifacts/models/xgb_model_latest.pkl",
    "../../artifacts/models/xgb_tfidf_vectorizer_latest.pkl"
)

evaluate(
    "SBERT + SVM",
    "../../artifacts/models/sbert_svm_model_latest.pkl",
    sbert=True
)

evaluate(
    "SBERT + Logistic Regression",
    "../../artifacts/models/sbert_lr_classifier_latest.pkl",
    sbert=True
)

# ============================
# Save Comparison Table
# ============================
df_results = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]
)

df_results = df_results.sort_values(
    by="F1-Score",
    ascending=False
).reset_index(drop=True)

df_results.to_csv(
    "../../artifacts/outputs/model_comparison_results_latest.csv",
    index=False
)

print("\nSaved comparison table → ../../artifacts/outputs/model_comparison_results_latest.csv")
print(df_results)