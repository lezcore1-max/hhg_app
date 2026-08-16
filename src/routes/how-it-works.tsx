import { createFileRoute } from "@tanstack/react-router";
import { PageShell } from "@/components/PageShell";
import { SectionCard } from "@/components/SectionCard";
import { chunking } from "@/lib/site-data";

export const Route = createFileRoute("/how-it-works")({
  head: () => ({
    meta: [
      { title: "How It Works · Four Chunkers, One Index" },
      {
        name: "description",
        content:
          "Inside Tilt: semantic, fixed-overlap, metadata-aware and sentence-window chunking over MSMARCO-XI, plus the guardrails that stop ungrounded answers.",
      },
      { property: "og:title", content: "How It Works · Four Chunkers, One Index" },
      {
        property: "og:description",
        content:
          "Multi-strategy chunking, hybrid retrieval, cross-encoder reranking and citation guardrails explained.",
      },
      { property: "og:type", content: "article" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: HowItWorksPage,
});

function HowItWorksPage() {
  return (
    <PageShell
      eyebrow="Under the hood"
      title="Four chunkers, one index."
      intro="Every passage is indexed four ways so retrieval can pick the lane that fits the question shape."
    >
      <div className="grid gap-4 md:grid-cols-2">
        {chunking.map((c, i) => (
          <SectionCard key={c.name} index={String(i + 1).padStart(2, "0")} title={c.name}>
            <p>{c.note}</p>
          </SectionCard>
        ))}
      </div>

      <div className="grain mt-6 rounded-lg border border-border bg-card p-6 sm:p-8">
        <p className="label-mono text-primary">Guardrails</p>
        <ul className="mt-4 space-y-3 text-sm leading-relaxed text-muted-foreground">
          <li>
            — Answers must cite at least one retrieved chunk; unsupported spans are
            stripped before speech synthesis.
          </li>
          <li>
            — Retrieval confidence below 0.55 returns{" "}
            <span className="text-primary">"I don't have that in the corpus"</span> instead
            of guessing.
          </li>
          <li>
            — Prompt-injection filter on transcribed text; no tool calls, no free-form web
            access.
          </li>
        </ul>
      </div>
    </PageShell>
  );
}
