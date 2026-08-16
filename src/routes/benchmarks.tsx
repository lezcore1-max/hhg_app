import { createFileRoute } from "@tanstack/react-router";
import { PageShell } from "@/components/PageShell";
import { budget, latency, quality } from "@/lib/site-data";

export const Route = createFileRoute("/benchmarks")({
  head: () => ({
    meta: [
      { title: "Benchmarks · 173ms P50 Voice RAG" },
      {
        name: "description",
        content:
          "Tilt latency percentiles across 1,000 spoken queries: 173ms P50, 214ms P95, with a full per-stage time budget plus groundedness, recall and WER.",
      },
      { property: "og:title", content: "Benchmarks · 173ms P50 Voice RAG" },
      {
        property: "og:description",
        content:
          "Percentiles, per-stage time budget and quality metrics from Tilt's 1,000-query eval harness.",
      },
      { property: "og:type", content: "article" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: BenchmarksPage,
});

function BenchmarksPage() {
  return (
    <PageShell
      eyebrow="Measured, not vibes"
      title="1,000 spoken queries."
      intro="Warm index, single Cloudflare region, mic-open to first spoken token."
    >
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {latency.map((l) => (
          <div key={l.k} className="grain rounded-lg border border-border bg-card p-6">
            <p className="label-mono text-primary">{l.k}</p>
            <p className="mt-3 font-display text-6xl text-card-foreground">{l.v}</p>
          </div>
        ))}
      </div>

      <div className="grain mt-6 rounded-lg border border-border bg-card p-6 sm:p-8">
        <p className="label-mono text-primary">Where the 173ms goes</p>
        <div className="mt-6 space-y-4">
          {budget.map((b) => (
            <div key={b.part}>
              <div className="flex items-baseline justify-between text-sm">
                <span className="text-card-foreground">{b.part}</span>
                <span className="label-mono text-primary">{b.ms}ms</span>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary"
                  style={{ width: `${b.pct}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        {quality.map((m) => (
          <div
            key={m.k}
            className="grain rounded-lg border border-border bg-secondary p-6"
          >
            <p className="label-mono text-primary">{m.k}</p>
            <p className="mt-3 font-display text-5xl">{m.v}</p>
          </div>
        ))}
      </div>
    </PageShell>
  );
}
