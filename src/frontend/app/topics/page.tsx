"use client";

import { useEffect, useState } from "react";
import { PageShell } from "@/components/page-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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

export default function TopicsPage() {
  const [topics, setTopics] = useState<Topic[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.topics().then((d) => setTopics(d.topics)).finally(() => setLoading(false));
  }, []);

  return (
    <PageShell>
      <div>
        <h1 className="text-2xl font-bold">Topic Explorer</h1>
        <p className="text-muted-foreground mt-1">
          10 latent topics discovered via LDA topic modelling
        </p>
      </div>

      {loading ? (
        <p className="text-muted-foreground">Loading topics…</p>
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
                {/* Weight bar for top word */}
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
