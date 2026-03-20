"use client"

import { useEffect, useState } from "react"
import { PageShell } from "@/components/page-shell"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { api, DatasetStats, ModelMetric, Trend } from "@/lib/api"
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts"
import { FileTextIcon, CheckCircleIcon, BrainCircuitIcon, TrendingUpIcon, AlertCircleIcon } from "lucide-react"

export function DashboardClient() {
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [metrics, setMetrics] = useState<ModelMetric[]>([])
  const [trends, setTrends] = useState<Trend[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    Promise.all([api.datasetStats(), api.metrics(), api.trends()])
      .then(([s, m, t]) => { setStats(s); setMetrics(m.models); setTrends(t.trends) })
      .catch(() => setError("Cannot reach the API. Make sure the backend is running on port 8000."))
      .finally(() => setLoading(false))
  }, [])

  const bestModel = metrics.reduce(
    (best, m) => (m.Accuracy > (best?.Accuracy ?? 0) ? m : best), metrics[0]
  )

  return (
    <PageShell>
      <div>
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground mt-1">Overview of the inflation research classification system</p>
      </div>

      {error && <ApiError message={error} />}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard icon={<FileTextIcon className="size-5 text-blue-500" />} label="Total Papers"
          value={loading ? null : stats?.total.toLocaleString() ?? "—"} />
        <StatCard icon={<CheckCircleIcon className="size-5 text-green-500" />} label="Relevant Papers"
          value={loading ? null : `${stats?.relevant.toLocaleString()} (${stats?.relevant_pct}%)`} />
        <StatCard icon={<BrainCircuitIcon className="size-5 text-purple-500" />} label="Models Trained"
          value={loading ? null : metrics.length.toString()} />
        <StatCard icon={<TrendingUpIcon className="size-5 text-orange-500" />} label="Best Accuracy"
          value={loading ? null : bestModel ? `${(bestModel.Accuracy * 100).toFixed(1)}%` : "—"}
          sub={bestModel?.Model} />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Publication Trends</CardTitle></CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-64 w-full" /> : (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={trends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="count" stroke="#6366f1" strokeWidth={2} dot={false} name="Papers" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Model Accuracy Comparison</CardTitle></CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-64 w-full" /> : (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={metrics} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0.8, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <YAxis type="category" dataKey="Model" width={130} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v: number) => `${(v * 100).toFixed(2)}%`} />
                  <Bar dataKey="Accuracy" fill="#6366f1" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>
    </PageShell>
  )
}

function StatCard({ icon, label, value, sub }: {
  icon: React.ReactNode; label: string; value: string | null; sub?: string
}) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center gap-3">
          {icon}
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground">{label}</p>
            {value === null
              ? <Skeleton className="h-7 w-24 mt-1" />
              : <p className="text-xl font-bold">{value}</p>}
            {sub && <p className="text-xs text-muted-foreground truncate">{sub}</p>}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function ApiError({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
      <AlertCircleIcon className="size-4 shrink-0" />
      {message}
    </div>
  )
}
