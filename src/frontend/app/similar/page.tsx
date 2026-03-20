"use client";

import { useState } from "react";
import { PageShell } from "@/components/page-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api, SimilarResult } from "@/lib/api";

export default function SimilarPage() {
  const [abstract, setAbstract] = useState("");
  const [topN, setTopN] = useState(5);
  const [results, setResults] = useState<SimilarResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searched, setSearched] = useState(false);

  async function handleSearch() {
    if (!abstract.trim()) return;
    setLoading(true);
    setError("");
    setResults([]);
    try {
      const data = await api.similar(abstract, topN);
      setResults(data.results);
      setSearched(true);
    } catch {
      setError("Failed to connect to the API. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <PageShell>
      <div>
        <h1 className="text-2xl font-bold">Similar Papers</h1>
        <p className="text-muted-foreground mt-1">
          Find the most semantically similar papers using SBERT embeddings
        </p>
      </div>

      {/* Input */}
      <Card>
        <CardContent className="pt-6 flex flex-col gap-4">
          <textarea
            className="w-full min-h-36 rounded-md border bg-background px-3 py-2 text-sm resize-y focus:outline-none focus:ring-2 focus:ring-ring"
            placeholder="Paste your research abstract here…"
            value={abstract}
            onChange={(e) => setAbstract(e.target.value)}
          />
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm text-muted-foreground whitespace-nowrap">Top N results</label>
              <input
                type="number"
                min={1}
                max={20}
                value={topN}
                onChange={(e) => setTopN(Number(e.target.value))}
                className="w-16 rounded-md border bg-background px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <Button
              className="ml-auto"
              onClick={handleSearch}
              disabled={loading || !abstract.trim()}
            >
              {loading ? "Searching…" : "Find Similar Papers"}
            </Button>
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {/* Results */}
      {searched && results.length === 0 && !loading && (
        <p className="text-muted-foreground text-center py-8">No results found.</p>
      )}

      <div className="flex flex-col gap-4">
        {results.map((r) => (
          <Card key={r.rank}>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-muted-foreground">#{r.rank}</span>
                  <Badge variant={r.label === 1 ? "default" : "destructive"}>
                    {r.label_text}
                  </Badge>
                </div>
                <Badge variant="outline" className="font-mono">
                  {(r.similarity_score * 100).toFixed(1)}% similar
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground leading-relaxed">{r.abstract_preview}</p>
              {r.doi && (
                <p className="text-xs text-muted-foreground mt-2 font-mono truncate">
                  DOI: {r.doi}
                </p>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </PageShell>
  );
}
