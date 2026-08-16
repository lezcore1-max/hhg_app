import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { MicOrb } from "@/components/MicOrb";
import { PageShell } from "@/components/PageShell";
import { askText, type AskResponse } from "@/lib/api";

export const Route = createFileRoute("/demo")({
  head: () => ({
    meta: [
      { title: "Live Demo · Tilt Voice RAG" },
      {
        name: "description",
        content:
          "Tap the mic and watch Tilt's voice RAG pipeline light up stage by stage with real measured latency: STT, retrieval, rerank and grounded generation.",
      },
      { property: "og:title", content: "Live Demo · Tilt Voice RAG" },
      {
        property: "og:description",
        content:
          "Run a sample spoken query through Tilt and see each pipeline stage report its measured latency.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: DemoPage,
});

function latencyBar(ms: number | undefined, max = 200) {
  if (!ms) return null;
  const pct = Math.min((ms / max) * 100, 100).toFixed(1);
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-border">
      <div
        className="h-full rounded-full bg-primary transition-all duration-700"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function StageRow({ label, ms }: { label: string; ms?: number }) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs text-muted-foreground label-mono">
        <span>{label}</span>
        <span className="text-primary font-bold">{ms != null ? `${ms} ms` : "—"}</span>
      </div>
      {latencyBar(ms)}
    </div>
  );
}

function DemoPage() {
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [textQuery, setTextQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const handleResult = (r: AskResponse) => {
    setResult(r);
    setError(null);
    if (r.transcript) setTextQuery(r.transcript);
  };

  const handleError = (msg: string) => {
    setError(msg);
    setResult(null);
  };

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!textQuery.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const r = await askText(textQuery.trim());
      setResult(r);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Error");
    } finally {
      setLoading(false);
    }
  };

  const totalMs =
    result?.total_pipeline_latency_ms ??
    result?.total_latency_ms ??
    result?.latency_ms;

  return (
    <PageShell
      eyebrow="Live pipeline"
      title="Hit the mic."
      intro="Speak your question — Sarvam AI transcribes it, RAG retrieves, reranks, and Gemini answers. All latencies are real."
    >
      {/* Mic Orb */}
      <div className="flex flex-col items-center gap-4">
        <MicOrb onResult={handleResult} onError={handleError} />
        <p className="label-mono text-xs text-muted-foreground">
          {result ? "Done! Tap again to ask another." : "Tap the orb to start speaking"}
        </p>
      </div>

      {/* OR type query */}
      <form
        onSubmit={handleTextSubmit}
        className="mt-8 flex w-full max-w-xl gap-2"
        id="text-query-form"
      >
        <input
          type="text"
          value={textQuery}
          onChange={(e) => setTextQuery(e.target.value)}
          placeholder="या यहाँ टाइप करें... (or type in Hindi)"
          className="flex-1 rounded-lg border border-border bg-card px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          id="demo-text-input"
        />
        <button
          type="submit"
          disabled={loading || !textQuery.trim()}
          className="label-mono rounded-lg bg-primary px-5 py-2.5 text-primary-foreground transition-transform hover:scale-105 disabled:opacity-50"
          id="demo-submit-btn"
        >
          {loading ? "..." : "पूछें"}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="mt-6 w-full max-w-xl rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          ⚠ {error}
        </div>
      )}

      {/* Result card */}
      {result && (
        <div className="mt-8 w-full max-w-xl space-y-5 rounded-xl border border-border bg-card p-6 grain">

          {/* Transcript */}
          {result.transcript && (
            <div>
              <p className="label-mono text-xs text-primary mb-1">Sarvam STT Transcript</p>
              <p className="text-sm text-muted-foreground italic">"{result.transcript}"</p>
            </div>
          )}

          {/* Answer */}
          <div>
            <p className="label-mono text-xs text-primary mb-1">
              Gemini Answer
              <span className="ml-2 text-muted-foreground normal-case font-normal">
                ({result.status})
              </span>
            </p>
            <p className="text-base leading-relaxed text-card-foreground">{result.answer}</p>
          </div>

          {/* Stage latencies */}
          <div>
            <p className="label-mono text-xs text-primary mb-3">Pipeline Latency</p>
            <div className="space-y-3">
              <StageRow label="Speech-to-text (Sarvam)" ms={result.stt_latency_ms} />
              <StageRow label="Hybrid retrieval" ms={result.retrieval_latency_ms} />
              <StageRow label="Cross-encoder rerank" ms={result.rerank_latency_ms} />
              <StageRow label="Grounded generation" ms={result.llm_latency_ms} />
              <div className="pt-1 border-t border-border flex justify-between text-sm font-bold">
                <span className="label-mono text-muted-foreground">Total</span>
                <span className="text-primary">{totalMs != null ? `${totalMs} ms` : "—"}</span>
              </div>
            </div>
          </div>

          {/* Top context */}
          {result.retrieved_context && result.retrieved_context.length > 0 && (
            <div>
              <p className="label-mono text-xs text-primary mb-2">Top Retrieved Context</p>
              <div className="space-y-2">
                {result.retrieved_context.slice(0, 2).map((doc, i) => (
                  <div key={i} className="rounded-lg bg-secondary/60 px-4 py-3 text-xs text-muted-foreground">
                    <p className="font-semibold text-foreground mb-1">Q: {doc.matched_question ?? doc.query}</p>
                    <p>A: {doc.answer}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Navigation */}
      <div className="mt-10 flex flex-wrap gap-3">
        <Link
          to="/benchmarks"
          className="label-mono rounded-full bg-primary px-6 py-3 text-primary-foreground transition-transform hover:scale-105"
        >
          See the numbers
        </Link>
        <Link
          to="/how-it-works"
          className="label-mono rounded-full border border-primary px-6 py-3 text-primary transition-colors hover:bg-primary hover:text-primary-foreground"
        >
          How it works
        </Link>
      </div>
    </PageShell>
  );
}
