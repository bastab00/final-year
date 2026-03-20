from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import warnings
import re
import string

warnings.filterwarnings("ignore")

import nltk
for _res in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(_res, quiet=True)
from nltk.corpus import stopwords as _sw
from nltk.stem import WordNetLemmatizer as _WNL

_stop_words = set(_sw.words("english"))
_lemmatizer = _WNL()

def _clean(text: str) -> str:
    """Apply same preprocessing used during training."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [_lemmatizer.lemmatize(w) for w in text.split() if w not in _stop_words]
    return " ".join(tokens)

BASE = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE / "artifacts" / "models"
DATA_DIR = BASE / "data" / "processed"
OUTPUTS_DIR = BASE / "artifacts" / "outputs"

app = FastAPI(title="Inflation Research Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_models: dict = {}
_sbert_clf = None
_dataset: pd.DataFrame = None
_sbert = None
_embeddings: np.ndarray = None

@app.on_event("startup")
def startup():
    global _models, _sbert_clf, _dataset, _sbert, _embeddings

    print("[INFO] Loading models...")
    _models = {
        "lr":  {"name": "Logistic Regression", "model": joblib.load(MODELS_DIR / "baseline_lr_model.pkl"),    "vectorizer": joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")},
        "svm": {"name": "SVM (Calibrated)",     "model": joblib.load(MODELS_DIR / "svm_calibrated_model.pkl"), "vectorizer": joblib.load(MODELS_DIR / "svm_tfidf_vectorizer.pkl")},
        "rf":  {"name": "Random Forest",         "model": joblib.load(MODELS_DIR / "rf_model.pkl"),             "vectorizer": joblib.load(MODELS_DIR / "rf_tfidf_vectorizer.pkl")},
        "xgb": {"name": "XGBoost",               "model": joblib.load(MODELS_DIR / "xgb_model.pkl"),            "vectorizer": joblib.load(MODELS_DIR / "xgb_tfidf_vectorizer.pkl")},
    }
    _sbert_clf = joblib.load(MODELS_DIR / "sbert_lr_classifier.pkl")

    print("[INFO] Loading dataset...")
    _dataset = pd.read_csv(DATA_DIR / "cleaned_inflation_dataset.csv")

    print("[INFO] Loading SBERT model...")
    try:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer(str(MODELS_DIR / "sbert_embedding_model"))

        cache_path = MODELS_DIR / "dataset_embeddings.npy"
        if cache_path.exists():
            _embeddings = np.load(str(cache_path))
            print(f"[INFO] Loaded cached embeddings ({_embeddings.shape[0]} abstracts)")
        else:
            print("[INFO] Computing SBERT embeddings for dataset (one-time, may take a few minutes)...")
            texts = _dataset["Abstract"].fillna("").tolist()
            _embeddings = _sbert.encode(texts, batch_size=64, show_progress_bar=True)
            np.save(str(cache_path), _embeddings)
            print("[INFO] Embeddings cached.")
    except Exception as e:
        print(f"[WARN] SBERT unavailable: {e}")

    print("[INFO] API ready.")


class PredictRequest(BaseModel):
    abstract: str


class SimilarRequest(BaseModel):
    abstract: str
    top_n: int = 5


@app.get("/")
def root():
    return {"status": "ok", "message": "Inflation Research Classifier API v1.0"}


@app.get("/health")
def health():
    return {
        "models_loaded": len(_models),
        "dataset_size": len(_dataset) if _dataset is not None else 0,
        "sbert_ready": _sbert is not None,
        "embeddings_ready": _embeddings is not None,
    }


@app.post("/predict")
def predict(body: PredictRequest):
    text = body.abstract.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Abstract cannot be empty")

    cleaned_text = _clean(text)
    results = []
    for key, m in _models.items():
        vec = m["vectorizer"].transform([cleaned_text])
        pred = int(m["model"].predict(vec)[0])
        proba = m["model"].predict_proba(vec)[0].tolist() if hasattr(m["model"], "predict_proba") else None

        results.append({
            "model_key": key,
            "model": m["name"],
            "prediction": pred,
            "label": "Relevant" if pred == 1 else "Not Relevant",
            "confidence": round(max(proba), 4) if proba else None,
            "prob_relevant": round(proba[1], 4) if proba else None,
            "prob_not_relevant": round(proba[0], 4) if proba else None,
        })

    if _sbert is not None and _sbert_clf is not None:
        emb = _sbert.encode([text])
        pred = int(_sbert_clf.predict(emb)[0])
        proba = _sbert_clf.predict_proba(emb)[0].tolist()
        results.append({
            "model_key": "sbert",
            "model": "SBERT + LR",
            "prediction": pred,
            "label": "Relevant" if pred == 1 else "Not Relevant",
            "confidence": round(max(proba), 4),
            "prob_relevant": round(proba[1], 4),
            "prob_not_relevant": round(proba[0], 4),
        })

    votes = sum(1 for r in results if r["prediction"] == 1)
    ensemble_pred = 1 if votes > len(results) / 2 else 0

    return {
        "abstract_preview": text[:300] + "..." if len(text) > 300 else text,
        "predictions": results,
        "ensemble": {
            "prediction": ensemble_pred,
            "label": "Relevant" if ensemble_pred == 1 else "Not Relevant",
            "votes_for": votes,
            "total_models": len(results),
        },
    }


@app.get("/metrics")
def metrics():
    df = pd.read_csv(OUTPUTS_DIR / "model_comparison_results.csv")
    records = df.round(4).to_dict(orient="records")
    return {"models": records}


@app.get("/dataset/stats")
def dataset_stats():
    df = _dataset
    return {
        "total": len(df),
        "relevant": int(df["Label"].sum()),
        "not_relevant": int((df["Label"] == 0).sum()),
        "relevant_pct": round(float(df["Label"].mean()) * 100, 1),
    }


@app.get("/bibliometrics/keywords")
def keywords(top_n: int = Query(default=20, ge=1, le=100)):
    df = pd.read_csv(DATA_DIR / "keyword_frequency.csv")
    return {"keywords": df.head(top_n).to_dict(orient="records")}


@app.get("/bibliometrics/authors")
def authors(top_n: int = Query(default=15, ge=1, le=50)):
    df = pd.read_csv(DATA_DIR / "top_authors.csv")
    return {"authors": df.head(top_n).to_dict(orient="records")}


@app.get("/bibliometrics/trends")
def trends():
    df = pd.read_csv(DATA_DIR / "dataset_with_years.csv")
    trend = (
        df.dropna(subset=["Year"])
        .groupby("Year")
        .size()
        .reset_index(name="count")
        .sort_values("Year")
    )
    trend["year"] = trend["Year"].astype(int)
    return {"trends": trend[["year", "count"]].to_dict(orient="records")}


@app.get("/bibliometrics/bigrams")
def bigrams(top_n: int = Query(default=20, ge=1, le=100)):
    df = pd.read_csv(DATA_DIR / "bigram_frequency.csv")
    return {"bigrams": df.head(top_n).to_dict(orient="records")}


@app.get("/topics")
def topics():
    import json
    topics_path = OUTPUTS_DIR / "lda_topics.json"
    if not topics_path.exists():
        raise HTTPException(status_code=503, detail="Topics data not available")
    with open(topics_path) as f:
        return json.load(f)


@app.post("/similar")
def similar(body: SimilarRequest):
    if _sbert is None or _embeddings is None:
        raise HTTPException(status_code=503, detail="SBERT model not available")

    top_n = max(1, min(body.top_n, 20))
    query_emb = _sbert.encode([body.abstract.strip()])

    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    normed_corpus = _embeddings / (norms + 1e-10)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    scores = (normed_corpus @ query_norm.T).flatten()

    top_idx = scores.argsort()[::-1][:top_n]

    results = []
    for rank, idx in enumerate(top_idx, 1):
        row = _dataset.iloc[int(idx)]
        abstract_raw = row.get("Abstract", "") or ""
        abstract_str = "" if str(abstract_raw) == "nan" else str(abstract_raw)
        results.append({
            "rank": rank,
            "doi": str(row.get("DOI", "") or ""),
            "abstract_preview": abstract_str[:300] + ("..." if len(abstract_str) > 300 else ""),
            "label": int(row["Label"]),
            "label_text": "Relevant" if row["Label"] == 1 else "Not Relevant",
            "similarity_score": round(float(scores[idx]), 4),
        })

    return {"query_preview": body.abstract[:200], "results": results}
