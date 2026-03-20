"use client";

import { useEffect, useState } from "react";
import { PageShell } from "@/components/page-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircleIcon } from "lucide-react";
import { api, ModelMetric } from "@/lib/api";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";

const METRICS = [
  { key: "Accuracy", color: "#6366f1" },
  { key: "Precision", color: "#22c55e" },
  { key: "Recall", color: "#f59e0b" },
  { key: "F1-Score", color: "#ec4899" },
];

export function ModelsClient() {
  const [models, setModels] = useState<ModelMetric[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    api.metrics()
      .then((d) => setModels(d.models))
      .catch(() => setError("Cannot reach the API. Make sure the backend is running on port 8000."))
      .finally(() => setLoading(false));
  }, []);

  const chartData = models.map((m) => ({
    name: m.Model.replace(" (Calibrated)", "").replace("Logistic ", "LR "),
    Accuracy: +(m.Accuracy * 100).toFixed(2),
    Precision: +(m.Precision * 100).toFixed(2),
    Recall: +(m.Recall * 100).toFixed(2),
    "F1-Score": +(m["F1-Score"] * 100).toFixed(2),
  }));

  return (
    <PageShell>
      <div>
        <h1 className="text-2xl font-bold">Model Comparison</h1>
        <p className="text-muted-foreground mt-1">
          Performance metrics across all trained classifiers
        </p>
      </div>

      {error && <ApiError message={error} />}

      <Card>
        <CardHeader>
          <CardTitle>Metrics Overview (%)</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <Skeleton className="h-80 w-full" />
          ) : (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[80, 100]} tickFormatter={(v) => `${v}%`} />
                <Tooltip formatter={(v: number) => `${v}%`} />
                <Legend />
                {METRICS.map((m) => (
                  <Bar key={m.key} dataKey={m.key} fill={m.color} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Full Results Table</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex flex-col gap-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-9 w-full" />
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 pr-4 font-medium">Model</th>
                    {METRICS.map((m) => (
                      <th key={m.key} className="text-right py-2 px-4 font-medium">{m.key}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => (
                    <tr key={m.Model} className="border-b last:border-0 hover:bg-muted/40">
                      <td className="py-2 pr-4 font-medium">{m.Model}</td>
                      <td className="text-right py-2 px-4">{(m.Accuracy * 100).toFixed(2)}%</td>
                      <td className="text-right py-2 px-4">{(m.Precision * 100).toFixed(2)}%</td>
                      <td className="text-right py-2 px-4">{(m.Recall * 100).toFixed(2)}%</td>
                      <td className="text-right py-2 px-4">{(m["F1-Score"] * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
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
