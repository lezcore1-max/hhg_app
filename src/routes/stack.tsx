import { createFileRoute, Link } from "@tanstack/react-router";
import { PageShell } from "@/components/PageShell";
import { stack } from "@/lib/site-data";

export const Route = createFileRoute("/stack")({
  head: () => ({
    meta: [
      { title: "Stack & Submission · Tilt Voice RAG" },
      {
        name: "description",
        content:
          "The tools behind Tilt: Sarvam STT, bge-small embeddings, Qdrant HNSW with BM25 hybrid, cross-encoder rerank, Groq Llama-3.1 on Cloudflare Workers.",
      },
      { property: "og:title", content: "Stack & Submission · Tilt Voice RAG" },
      {
        property: "og:description",
        content:
          "Every component of the Tilt voice RAG pipeline, plus repo and dataset links for Hacker House Goa reviewers.",
      },
      { property: "og:type", content: "article" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: StackPage,
});

function StackPage() {
  return (
    <PageShell
      eyebrow="Built with"
      title="The stack."
      intro="Small models, tight hops, no orchestration framework in the hot path."
    >
      <div className="flex flex-wrap gap-3">
        {stack.map((s) => (
          <span
            key={s}
            className="label-mono rounded-full border border-primary/60 px-4 py-2 text-primary"
          >
            {s}
          </span>
        ))}
      </div>

      <div className="grain mt-14 rounded-lg border border-border bg-card p-8 text-center sm:p-12">
        <h2 className="text-4xl sm:text-5xl">Submitted for Hacker House Goa.</h2>
        <p className="mx-auto mt-4 max-w-lg text-sm leading-relaxed text-muted-foreground">
          Repo, eval harness, latency logs and a 90-second walkthrough — all in one place.
          Reviewers, start with the demo.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer noopener"
            className="label-mono rounded-full bg-primary px-6 py-3 text-primary-foreground transition-transform hover:scale-105"
          >
            GitHub repo
          </a>
          <a
            href="https://huggingface.co/datasets/ai4bharat/MSMARCO-XI"
            target="_blank"
            rel="noreferrer noopener"
            className="label-mono rounded-full border border-primary px-6 py-3 text-primary transition-colors hover:bg-primary hover:text-primary-foreground"
          >
            Dataset
          </a>
          <Link
            to="/demo"
            className="label-mono rounded-full border border-primary px-6 py-3 text-primary transition-colors hover:bg-primary hover:text-primary-foreground"
          >
            Run the demo
          </Link>
        </div>
      </div>
    </PageShell>
  );
}
