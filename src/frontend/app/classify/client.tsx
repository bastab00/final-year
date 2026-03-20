"use client";

import { useState } from "react";
import { PageShell } from "@/components/page-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AlertCircleIcon } from "lucide-react";
import { api, PredictResponse } from "@/lib/api";

export function ClassifyClient() {
  const [abstract, setAbstract] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit() {
    if (!abstract.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await api.predict(abstract);
      setResult(data);
    } catch {
      setError("Failed to connect to the API. Make sure the backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <PageShell>
      <div>
        <h1 className="text-2xl font-bold">Abstract Classifier</h1>
        <p className="text-muted-foreground mt-1">
          Paste a research abstract to classify it across all models
        </p>
      </div>

      <Card>
        <CardContent className="pt-6 flex flex-col gap-4">
          <textarea
            className="w-full min-h-40 rounded-md border bg-background px-3 py-2 text-sm resize-y focus:outline-none focus:ring-2 focus:ring-ring"
            placeholder="Paste your research abstract here…"
            value={abstract}
            onChange={(e) => setAbstract(e.target.value)}
          />
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">{abstract.length} characters</span>
            <Button onClick={handleSubmit} disabled={loading || !abstract.trim()}>
              {loading ? "Classifying…" : "Classify Abstract"}
            </Button>
          </div>
          {error && <ApiError message={error} />}
        </CardContent>
      </Card>

      {result && (
        <>
          <Card className={result.ensemble.prediction === 1 ? "border-green-500" : "border-red-400"}>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Ensemble Verdict</p>
                  <p className="text-2xl font-bold">{result.ensemble.label}</p>
                  <p className="text-sm text-muted-foreground">
                    {result.ensemble.votes_for} of {result.ensemble.total_models} models agree
                  </p>
                </div>
                <Badge
                  className="text-lg px-4 py-2"
                  variant={result.ensemble.prediction === 1 ? "default" : "destructive"}
                >
                  {result.ensemble.prediction === 1 ? "Relevant" : "Not Relevant"}
                </Badge>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {result.predictions.map((p) => (
              <Card key={p.model_key}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium">{p.model}</CardTitle>
                    <Badge variant={p.prediction === 1 ? "default" : "destructive"}>
                      {p.label}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="flex flex-col gap-3">
                  {p.confidence !== null && (
                    <div>
                      <div className="flex justify-between text-xs text-muted-foreground mb-1">
                        <span>Confidence</span>
                        <span>{(p.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${p.prediction === 1 ? "bg-green-500" : "bg-red-400"}`}
                          style={{ width: `${p.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                  {p.prob_relevant !== null && (
                    <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                      <div className="rounded bg-muted px-2 py-1">
                        <span className="block font-medium text-foreground">
                          {(p.prob_relevant * 100).toFixed(1)}%
                        </span>
                        Relevant
                      </div>
                      <div className="rounded bg-muted px-2 py-1">
                        <span className="block font-medium text-foreground">
                          {(p.prob_not_relevant! * 100).toFixed(1)}%
                        </span>
                        Not Relevant
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </>
      )}
    </PageShell>
  );
}

function ApiError({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
      <AlertCircleIcon className="size-4 shrink-0" />
      {message}
    </div>
  );
}
