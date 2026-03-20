# inflation research classifier

a full-stack app that classifies academic abstracts as relevant or not relevant to inflation research, using a set of trained ml models.

## what it does

- classifies research abstracts using 5 models: logistic regression, svm, random forest, xgboost, and sbert+lr
- returns an ensemble verdict with per-model confidence scores
- finds semantically similar papers from the dataset using sbert embeddings
- provides bibliometric analysis: top keywords, authors, publication trends, bigrams
- visualises lda topic clusters discovered from the corpus

## stack

- **backend** - fastapi, scikit-learn, xgboost, sentence-transformers
- **frontend** - next.js 14, shadcn/ui, tailwind css, recharts

## getting started

### backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.app.api:app --reload
```

runs on `http://localhost:8000`

### frontend

```bash
cd src/frontend
cp .env.local.example .env.local
bun install
bun dev
```

runs on `http://localhost:3000`

## dataset

1,134 research paper abstracts labeled as relevant (234) or not relevant (900) to inflation research.

## authors

avhi, bastab, lakhya, sibga


