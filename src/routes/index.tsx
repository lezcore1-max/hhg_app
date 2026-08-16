import { createFileRoute, Link } from "@tanstack/react-router";
import heroSunrise from "@/assets/hero-sunrise.jpg";
import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";
import { latency, nav } from "@/lib/site-data";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Tilt · Voice-Native RAG in 173ms" },
      {
        name: "description",
        content:
          "Tilt is a voice-native RAG engine: speak a question, get a grounded answer in 173ms P50. Multi-strategy chunking on MSMARCO-XI, Sarvam STT, guardrailed generation. Built for Hacker House Goa 2026.",
      },
      { property: "og:title", content: "Tilt · Voice-Native RAG in 173ms" },
      {
        property: "og:description",
        content:
          "Speak. Retrieve. Answer. A sub-200ms voice RAG pipeline with hybrid chunking and hallucination guardrails — my Hacker House Goa 2026 submission.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: HomePage,
});

const blurbs: Record<string, string> = {
  "/demo": "Tap the mic, watch every stage report its latency.",
  "/how-it-works": "Four chunking strategies, hybrid retrieval, guardrails.",
  "/benchmarks": "Percentiles and per-stage budget from 1,000 queries.",
  "/stack": "Every model and service in the hot path.",
};

function HomePage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <SiteHeader />

      <section className="grain relative overflow-hidden">
        <img
          src={heroSunrise}
          alt="Retro screen-print sunrise over palms and ocean waves"
          width={1536}
          height={1024}
          className="absolute inset-0 h-full w-full object-cover opacity-30"
        />
        <div className="relative mx-auto max-w-6xl px-5 py-20 sm:py-28">
          <p className="label-mono text-primary">HH Goa 2026 · Task 02 submission</p>
          <h1 className="mt-5 text-6xl tracking-tight sm:text-8xl lg:text-9xl">
            Speak.
            <br />
            Retrieve.
            <br />
            <span className="text-primary">173ms.</span>
          </h1>
          <p className="mt-8 max-w-xl text-sm leading-relaxed text-muted-foreground sm:text-base">
            Tilt is a voice-native RAG engine. Ask out loud and it transcribes, searches
            8.8M MSMARCO-XI passages with four chunking strategies, reranks, and speaks
            back a grounded answer — median end-to-end 173 milliseconds.
          </p>
          <div className="mt-10 flex flex-wrap gap-3">
            <Link
              to="/demo"
              className="label-mono rounded-full bg-primary px-6 py-3 text-primary-foreground transition-transform hover:scale-105"
            >
              Try the demo
            </Link>
            <Link
              to="/benchmarks"
              className="label-mono rounded-full border border-primary px-6 py-3 text-primary transition-colors hover:bg-primary hover:text-primary-foreground"
            >
              See the numbers
            </Link>
          </div>
        </div>
      </section>

      <div className="overflow-hidden border-y border-border bg-primary py-3">
        <div className="ticker-track">
          {Array.from({ length: 2 }).map((_, i) => (
            <span key={i} className="label-mono flex shrink-0 text-primary-foreground">
              {Array.from({ length: 6 }).map((__, j) => (
                <span key={j} className="px-6">
                  #RAGInGoa · 173ms P50 · 94.2% groundedness · zero hallucinations shipped
                  ·
                </span>
              ))}
            </span>
          ))}
        </div>
      </div>

      <section className="mx-auto max-w-6xl px-5 py-20 sm:py-28">
        <p className="label-mono text-primary">Latency at a glance</p>
        <h2 className="mt-4 max-w-2xl text-5xl sm:text-6xl">Fast, at every percentile.</h2>
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {latency.map((l) => (
            <div key={l.k} className="grain rounded-lg border border-border bg-card p-6">
              <p className="label-mono text-primary">{l.k}</p>
              <p className="mt-3 font-display text-6xl text-card-foreground">{l.v}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="border-t border-border bg-secondary/60">
        <div className="mx-auto max-w-6xl px-5 py-20 sm:py-28">
          <p className="label-mono text-primary">Explore</p>
          <h2 className="mt-4 max-w-2xl text-5xl sm:text-6xl">Dig in.</h2>
          <div className="mt-10 grid gap-4 md:grid-cols-2">
            {nav.map((n) => (
              <Link
                key={n.to}
                to={n.to}
                className="grain group rounded-lg border border-border bg-card p-6 transition-transform duration-300 hover:-translate-y-1 sm:p-8"
              >
                <p className="label-mono text-primary">{n.label}</p>
                <p className="mt-3 text-3xl text-card-foreground sm:text-4xl">
                  {blurbs[n.to]}
                </p>
                <span className="label-mono mt-5 inline-block text-primary">
                  Open →
                </span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <SiteFooter />
    </div>
  );
}
