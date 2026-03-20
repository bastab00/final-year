"use client";

import { useEffect, useState } from "react";
import { PageShell } from "@/components/page-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api, Keyword, Author, Trend, Bigram } from "@/lib/api";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from "recharts";

export default function BibliometricsPage() {
  const [keywords, setKeywords] = useState<Keyword[]>([]);
  const [authors, setAuthors] = useState<Author[]>([]);
  const [trends, setTrends] = useState<Trend[]>([]);
  const [bigrams, setBigrams] = useState<Bigram[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.keywords(20), api.authors(15), api.trends(), api.bigrams(20)])
      .then(([k, a, t, b]) => {
        setKeywords(k.keywords);
        setAuthors(a.authors);
        setTrends(t.trends);
        setBigrams(b.bigrams);
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <PageShell>
      <div>
        <h1 className="text-2xl font-bold">Bibliometrics</h1>
        <p className="text-muted-foreground mt-1">
          Analysis of keywords, authors, and publication trends
        </p>
      </div>

      <Tabs defaultValue="keywords">
        <TabsList>
          <TabsTrigger value="keywords">Keywords</TabsTrigger>
          <TabsTrigger value="authors">Authors</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
          <TabsTrigger value="bigrams">Bigrams</TabsTrigger>
        </TabsList>

        <TabsContent value="keywords" className="mt-4">
          <Card>
            <CardHeader><CardTitle>Top 20 Keywords</CardTitle></CardHeader>
            <CardContent>
              {loading ? <Loader /> : (
                <ResponsiveContainer width="100%" height={420}>
                  <BarChart data={keywords} layout="vertical" margin={{ left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="Keyword" tick={{ fontSize: 12 }} width={90} />
                    <Tooltip />
                    <Bar dataKey="Frequency" fill="#6366f1" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="authors" className="mt-4">
          <Card>
            <CardHeader><CardTitle>Top 15 Authors by Paper Count</CardTitle></CardHeader>
            <CardContent>
              {loading ? <Loader /> : (
                <ResponsiveContainer width="100%" height={420}>
                  <BarChart data={authors} layout="vertical" margin={{ left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" allowDecimals={false} />
                    <YAxis type="category" dataKey="Author_Standardized" tick={{ fontSize: 12 }} width={110} />
                    <Tooltip />
                    <Bar dataKey="Paper_Count" fill="#22c55e" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trends" className="mt-4">
          <Card>
            <CardHeader><CardTitle>Publications Per Year</CardTitle></CardHeader>
            <CardContent>
              {loading ? <Loader /> : (
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={trends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Line type="monotone" dataKey="count" stroke="#6366f1" strokeWidth={2} dot={{ r: 4 }} name="Papers" />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="bigrams" className="mt-4">
          <Card>
            <CardHeader><CardTitle>Top 20 Bigrams</CardTitle></CardHeader>
            <CardContent>
              {loading ? <Loader /> : (
                <ResponsiveContainer width="100%" height={420}>
                  <BarChart data={bigrams} layout="vertical" margin={{ left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="Bigram" tick={{ fontSize: 11 }} width={130} />
                    <Tooltip />
                    <Bar dataKey="Frequency" fill="#f59e0b" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </PageShell>
  );
}

function Loader() {
  return <div className="h-64 flex items-center justify-center text-muted-foreground">Loading…</div>;
}
