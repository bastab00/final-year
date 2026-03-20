const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

export interface Prediction {
  model_key: string;
  model: string;
  prediction: number;
  label: string;
  confidence: number | null;
  prob_relevant: number | null;
  prob_not_relevant: number | null;
}

export interface PredictResponse {
  abstract_preview: string;
  predictions: Prediction[];
  ensemble: {
    prediction: number;
    label: string;
    votes_for: number;
    total_models: number;
  };
}

export interface ModelMetric {
  Model: string;
  Accuracy: number;
  Precision: number;
  Recall: number;
  "F1-Score": number;
}

export interface DatasetStats {
  total: number;
  relevant: number;
  not_relevant: number;
  relevant_pct: number;
}

export interface Keyword {
  Keyword: string;
  Frequency: number;
}

export interface Author {
  Author_Standardized: string;
  Paper_Count: number;
}

export interface Trend {
  year: number;
  count: number;
}

export interface Bigram {
  Bigram: string;
  Frequency: number;
}

export interface Topic {
  topic_id: number;
  top_words: { word: string; weight: number }[];
}

export interface SimilarResult {
  rank: number;
  doi: string;
  abstract_preview: string;
  label: number;
  label_text: string;
  similarity_score: number;
}

export const api = {
  health: () => apiFetch<{ models_loaded: number; dataset_size: number; sbert_ready: boolean }>("/health"),

  datasetStats: () => apiFetch<DatasetStats>("/dataset/stats"),

  predict: (abstract: string) =>
    apiFetch<PredictResponse>("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ abstract }),
    }),

  metrics: () => apiFetch<{ models: ModelMetric[] }>("/metrics"),

  keywords: (top_n = 20) =>
    apiFetch<{ keywords: Keyword[] }>(`/bibliometrics/keywords?top_n=${top_n}`),

  authors: (top_n = 15) =>
    apiFetch<{ authors: Author[] }>(`/bibliometrics/authors?top_n=${top_n}`),

  trends: () => apiFetch<{ trends: Trend[] }>("/bibliometrics/trends"),

  bigrams: (top_n = 20) =>
    apiFetch<{ bigrams: Bigram[] }>(`/bibliometrics/bigrams?top_n=${top_n}`),

  topics: () => apiFetch<{ num_topics: number; topics: Topic[] }>("/topics"),

  similar: (abstract: string, top_n = 5) =>
    apiFetch<{ query_preview: string; results: SimilarResult[] }>("/similar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ abstract, top_n }),
    }),
};
