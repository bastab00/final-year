import type { Metadata } from "next"
export const metadata: Metadata = { title: "Dashboard — InflationAI" }

// ↑ server-side metadata, page itself is a client component
export { DashboardClient as default } from "./client"
