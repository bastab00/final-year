"use client";

import { useEffect, useState } from "react";
import { PageShell } from "@/components/page-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircleIcon } from "lucide-react";
import { api, Topic } from "@/lib/api";

const TOPIC_LABELS: Record<number, string> = {
  0: "Research Methods",
  1: "Forecasting",
  2: "Exchange Rates & Growth",
  3: "Monetary Policy",
  4: "Prices & Factors",
  5: "Financial Risk",
  6: "Price Indices",
  7: "Central Banking",
  8: "Market Expectations",
  9: "Energy & Yields",
};

export function TopicsClient() {
  const [topics, setTopics] = useState<Topic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    api.topics()
      .then((d) => setTopics(d.topics))
      .catch(() => setError("Cannot reach the API. Make sure the backend is running on port 8000."))
      .finally(() => setLoading(false));
  }, []);

  return (
    <PageShell>
      <div>
        <h1 className="text-2xl font-bold">Topic Explorer</h1>
        <p className="text-muted-foreground mt-1">
          10 latent topics discovered via LDA topic modelling
        </p>
      </div>

      {error && <ApiError message={error} />}

      {loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 10 }).map((_, i) => (
            <Skeleton key={i} className="h-48 w-full rounded-xl" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {topics.map((topic) => (
            <Card key={topic.topic_id}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-semibold">
                    {TOPIC_LABELS[topic.topic_id] ?? `Topic ${topic.topic_id}`}
                  </CardTitle>
                  <Badge variant="outline">#{topic.topic_id}</Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {topic.top_words.slice(0, 8).map(({ word, weight }) => (
                    <span
                      key={word}
                      className="rounded-full bg-muted px-2 py-0.5 text-xs font-medium"
                      title={`Weight: ${weight.toFixed(4)}`}
                    >
                      {word}
                    </span>
                  ))}
                </div>
                <div className="mt-3">
                  {topic.top_words.slice(0, 5).map(({ word, weight }) => (
                    <div key={word} className="flex items-center gap-2 mb-1">
                      <span className="text-xs w-20 truncate text-muted-foreground">{word}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full rounded-full bg-indigo-500"
                          style={{ width: `${(weight / topic.top_words[0].weight) * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-12 text-right">
                        {(weight * 100).toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
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
