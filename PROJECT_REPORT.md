# Final Year Project — Inflation Research Topic Classification and Analysis

**Abstract**
- **Summary:**: This project builds an end-to-end system for analyzing, modeling, and visualizing topical trends in academic and policy literature about inflation. The pipeline covers data collection, cleaning, feature extraction (including sentence embeddings and topic models), supervised classification using multiple ML approaches, explainability, and a small web API for model inference and result delivery.

**Project Objectives**
- **Primary Goal:**: Classify documents related to inflation into research topics and analyze topic trends over time.
- **Secondary Goals:**: Compare classical, embedding-based, and transformer-based classifiers; provide interpretable explanations of model behavior; produce visualizations and exported artifacts suitable for inclusion in a final-year report.

**Motivation**
- **Context:**: Inflation-related literature is large and heterogeneous. Automated topic detection and classification help researchers and policymakers track trends, identify influential authors, and extract key themes over time.

**Contributions**
- **End-to-end Pipeline:**: A reproducible pipeline from raw data through preprocessing, modeling, evaluation, and visualization.
- **Multiple Model Families:**: Baselines (bag-of-words), classical ML (SVM, Random Forest, XGBoost), embedding + classifiers (SBERT + classifier), and BERT-style models.
- **Explainability & Artifacts:**: SHAP explanations, topic model outputs, interactive HTML visualizations, and saved model artifacts for reuse.

**Repository Overview**
- **Root:**: Top-level project files including [README.md](README.md) and [requirements.txt](requirements.txt).
- **Data:**: Raw and processed datasets: [data/raw](data/raw) and [data/processed](data/processed).
- **Scripts & Source:**: Main code under [src](src) including preprocessing, models, pipeline, and API.
- **Artifacts:**: Trained models and embeddings under [artifacts/models](artifacts/models) and [artifacts/models/sbert_embedding_model](artifacts/models/sbert_embedding_model).
- **Outputs:**: Evaluation CSVs, topic JSON, and HTML visualizations in [outputs](outputs).

**Important Files and Locations**
- **Preprocessing:**: [src/preprocessing/preprocess_dataset.py](src/preprocessing/preprocess_dataset.py), [src/preprocessing/text_cleaner.py](src/preprocessing/text_cleaner.py), [src/preprocessing/extract_author.py](src/preprocessing/extract_author.py)
- **Feature engineering & topic modeling:**: [src/topic_modeling](src/topic_modeling) (topic modeling scripts), [src/bibliometrics/bigrams_top_keywords.py](src/bibliometrics/bigrams_top_keywords.py)
- **Model training:**: [src/models/train_baseline_classifier.py](src/models/train_baseline_classifier.py), [src/models/train_bert_classifier.py](src/models/train_bert_classifier.py), [src/models/train_sbert_classifier.py](src/models/train_sbert_classifier.py), [src/models/train_gaussian_svm.py](src/models/train_gaussian_svm.py), [src/models/train_random_forest.py](src/models/train_random_forest.py), [src/models/train_xgboost_classifier.py](src/models/train_xgboost_classifier.py)
- **Pipeline runner:**: [src/pipeline/run_pipeline.py](src/pipeline/run_pipeline.py)
- **API:**: [src/app/api.py](src/app/api.py)
- **Explainability:**: [src/explainability/shap_explain.py](src/explainability/shap_explain.py)

**Dataset**
- **Sources:**: The raw data files are located in [data/raw](data/raw). Notable raw files include [data/raw/inflation_dataset.json](data/raw/inflation_dataset.json) and [data/raw/inflation_dataset_latest_version.csv](data/raw/inflation_dataset_latest_version.csv).
- **Processed Data:**: Preprocessed and cleaned datasets are under [data/processed](data/processed), for example [data/processed/cleaned_inflation_dataset.csv](data/processed/cleaned_inflation_dataset.csv) and [data/processed/dataset_latest_version.csv](data/processed/dataset_latest_version.csv).
- **Fields & Schema:**: Typical document records include fields such as title, abstract/text, authors, publication year, venue, and any topic labels used for supervised training. Exact column names and transformations are defined in [src/preprocessing/preprocess_dataset.py](src/preprocessing/preprocess_dataset.py).

**Preprocessing**
- **Cleaning Steps:**: The preprocessing pipeline performs:
  - **Load & merge:**: Merge different raw sources and incremental updates using scripts in [src/preprocessing](src/preprocessing).
  - **Text cleaning:**: Lowercasing, punctuation removal, whitespace normalization, and optional language-specific normalization implemented in [src/preprocessing/text_cleaner.py](src/preprocessing/text_cleaner.py).
  - **Tokenization & lemmatization:**: Tokenization and lemmatization (where applicable) to reduce inflectional variance.
  - **Stopword removal & n-grams:**: Removal of common stopwords and construction of bigrams; see [src/bibliometrics/bigrams_top_keywords.py](src/bibliometrics/bigrams_top_keywords.py).
  - **Deduplication & filtering:**: Remove duplicate records and filter by language and minimum content length.
- **Saved outputs:**: Cleaned datasets in [data/processed](data/processed) and intermediate frequency tables in [data/processed/bigram_frequency.csv](data/processed/bigram_frequency.csv) and [data/processed/keyword_frequency.csv](data/processed/keyword_frequency.csv).

**Feature Engineering**
- **Bag-of-Words / TF-IDF:**: Classical feature matrices produced for baseline models.
- **Embeddings:**: Sentence embeddings are produced using SBERT models and saved under [artifacts/models/sbert_embedding_model](artifacts/models/sbert_embedding_model) and dataset embeddings at [artifacts/models/dataset_embeddings.npy](artifacts/models/dataset_embeddings.npy).
- **Topic Features:**: Unsupervised topic distributions (e.g., LDA) are stored in [outputs/lda_topics.json](outputs/lda_topics.json) and visualized in [outputs/ml_topics_over_time.html](outputs/ml_topics_over_time.html).

**Topic Modeling & Bibliometrics**
- **LDA & Topic Trends:**: Topic modeling scripts under [src/topic_modeling](src/topic_modeling) generate the LDA model and export topics to [outputs/lda_topics.json]. Trend visualizations (topic prevalence over time) are exported as [outputs/ml_topics_over_time.html] and [outputs/nonml_topics_over_time.html].
- **Author Analysis:**: Author-level bibliometrics (top authors, co-authorship graphs) implemented in [src/bibliometrics/graph_top_authors.py](src/bibliometrics/graph_top_authors.py) and [src/bibliometrics/top_authors.py](src/bibliometrics/top_authors.py). Output top authors list is [data/processed/top_authors.csv](data/processed/top_authors.csv).

**Modeling Approaches**
- **Baselines:**: Simple baselines using bag-of-words features in [src/models/train_baseline_classifier.py](src/models/train_baseline_classifier.py).
- **Classical ML:**: SVMs, Random Forests, Gaussian Naive Bayes / Gaussian SVM in [src/models/train_gaussian_svm.py](src/models/train_gaussian_svm.py) and [src/models/train_random_forest.py](src/models/train_random_forest.py).
- **Embedding-based:**: SBERT embeddings + classifier in [src/models/train_sbert_classifier.py](src/models/train_sbert_classifier.py) and logistic-regression on embeddings in [src/models/train_sbert_lr_classifier.py](src/models/train_sbert_lr_classifier.py).
- **Transformer-based:**: Fine-tuned BERT-style classifier in [src/models/train_bert_classifier.py](src/models/train_bert_classifier.py).
- **Ensembles & Comparisons:**: Model comparison and selection utilities output CSVs such as [outputs/model_comparison_results_latest.csv](outputs/model_comparison_results_latest.csv) and related versions.

**Training Workflow**
- **Entry point:**: Orchestrate the full workflow via [src/pipeline/run_pipeline.py](src/pipeline/run_pipeline.py).
- **Common steps:**: Split data into train/validation/test, preprocess text, compute features/embeddings, train candidate models, evaluate on validation folds, pick best models, run final evaluation on test set.
- **Reproducibility:**: All training scripts accept seed parameters where possible; ensure to set seeds in configuration to reproduce runs.

**Evaluation Metrics & Equations**
- **Metrics used:**: Accuracy, Precision, Recall, F1-score, ROC-AUC for binary/multiclass tasks (as applicable). Saved evaluation outputs are in [outputs](outputs).
- **Formulas:**:
  - **Accuracy:**: $\\text{Accuracy} = \\dfrac{TP + TN}{TP + TN + FP + FN}$
  - **Precision:**: $\\text{Precision} = \\dfrac{TP}{TP + FP}$
  - **Recall:**: $\\text{Recall} = \\dfrac{TP}{TP + FN}$
  - **F1-score:**: $\\text{F1} = 2 \\times \\dfrac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$

**Explainability**
- **SHAP:**: Per-model explanation support via [src/explainability/shap_explain.py](src/explainability/shap_explain.py). Example output for an SVM model is [outputs/svm_shap_explanation.html](outputs/svm_shap_explanation.html).
- **Interpretation Guidance:**: Use SHAP plots to identify important features/terms and inspect example-level explanations for misclassified documents.

**Artifacts & Outputs**
- **Trained models & embeddings:**: [artifacts/models](artifacts/models) and [artifacts/models/dataset_embeddings.npy](artifacts/models/dataset_embeddings.npy).
- **Evaluation CSVs:**: [outputs/model_comparison_results_latest.csv](outputs/model_comparison_results_latest.csv), [outputs/model_comparison_results_v2.csv](outputs/model_comparison_results_v2.csv), etc.
- **Topic JSON & Visuals:**: [outputs/lda_topics.json](outputs/lda_topics.json), [outputs/ml_topics_over_time.html](outputs/ml_topics_over_time.html).
- **Explainability HTMLs:**: [outputs/svm_shap_explanation.html](outputs/svm_shap_explanation.html).

**API & Deployment**
- **Lightweight API:**: A FastAPI application exposes endpoints for health checks and model predictions at [src/app/api.py](src/app/api.py).
- **Run locally:**: Start the API with `uvicorn src.app.api:app --reload` (example below).

**Usage & Reproduction Instructions**
- **Prerequisites:**:
  - **Python:**: 3.8+ recommended.
  - **Virtualenv:**: Create and activate a virtual environment.
- **Install dependencies:**:
```bash
python -m venv venv
# Windows
venv\\Scripts\\activate
# Unix/macOS
source venv/bin/activate
pip install -r requirements.txt
```
- **Preprocess data:**: Run the preprocessing script to produce cleaned datasets.
```bash
python src/preprocessing/preprocess_dataset.py
```
- **Run the pipeline (end-to-end):**:
```bash
python src/pipeline/run_pipeline.py
```
- **Train a specific model:**: Example training script for SBERT-based classifier:
```bash
python src/models/train_sbert_classifier.py --config configs/sbert_config.yml
```
- **Start API:**:
```bash
uvicorn src.app.api:app --reload
```
- **Quick API test:**: Example Python snippet to test endpoints (adjust to your local port):
```python
import urllib.request, json
req = urllib.request.Request('http://localhost:8000/predict', data=json.dumps({}).encode('utf-8'), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as r:
    print(r.read().decode())
```

**Running Experiments & Notes**
- **Hardware:**: For transformer fine-tuning, use a GPU-enabled environment where possible.
- **Hyperparameters:**: Hyperparameter choices (learning rate, epochs, batch size, regularization) are controlled in the individual training scripts or config files — check model-specific scripts under [src/models](src/models).
- **Logging & Checkpoints:**: Training scripts write checkpoints to [artifacts/models] and logs to the outputs folder when configured.

**Results Summary (Where to find them)**
- **Model comparisons:**: See [outputs/model_comparison_results_latest.csv](outputs/model_comparison_results_latest.csv) and other versions for detailed per-model metrics and runtime stats.
- **Topic outputs & visualizations:**: [outputs/lda_topics.json](outputs/lda_topics.json) and [outputs/ml_topics_over_time.html](outputs/ml_topics_over_time.html).
- **Explainability artifacts:**: [outputs/svm_shap_explanation.html](outputs/svm_shap_explanation.html) demonstrates feature importance for a selected model.

**Limitations & Ethical Considerations**
- **Label quality:**: Model performance depends heavily on the quality of topic labels and data coverage. Manually curated labels or weak supervision can introduce biases.
- **Data bias & representativeness:**: The dataset may overrepresent certain venues, years, or regions; interpret results with that context in mind.
- **Explainability limits:**: Feature-level explanations (e.g., SHAP) are informative but not definitive causal statements.

**Conclusions**
- **Summary:**: The project delivers a reproducible system for classifying inflation literature, comparing modeling approaches, and producing explainable outputs and visualizations suitable for reporting and further analysis.
- **Takeaway:**: Embedding-based and transformer-based methods tend to be stronger on semantic generalization; classical models are often faster and more interpretable with careful feature engineering.

**Future Work**
- **Expand data sources:**: Include additional corpora, cross-lingual sources, and full-text where available.
- **Semi-supervised labeling:**: Use active learning or weak supervision to expand labeled data with minimal annotation cost.
- **Temporal models:**: Explore topic evolution with dynamic topic models or sequence models that explicitly model time.
- **Interactive front-end:**: Enhance the frontend visualizations for interactive exploration and filtering by author, year, and venue.

**References & Suggested Citations**
- **Libraries & Tools:**: scikit-learn, transformers, sentence-transformers, gensim, fastapi, uvicorn, SHAP.
- **Methodology references:**: Cite standard papers for LDA, SBERT, BERT, and SHAP in the final report.

**Appendix: File Map (quick links)**
- **Top-level:**: [README.md](README.md), [requirements.txt](requirements.txt)
- **Scripts:**: [scripts](scripts)
- **Source:**: [src/preprocessing](src/preprocessing), [src/models](src/models), [src/pipeline](src/pipeline), [src/explainability](src/explainability), [src/app/api.py](src/app/api.py)
- **Data:**: [data/raw](data/raw), [data/processed](data/processed)
- **Artifacts & outputs:**: [artifacts/models](artifacts/models), [outputs](outputs)

**Contact & Author**
- **Project owner:**: See repository metadata or project header for author information. For questions about reproducing results or report generation, run the pipeline and consult the scripts referenced above.


---

*Generated on request. The file excludes any references to private API keys or the script named prebict as requested.*
