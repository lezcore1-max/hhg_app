export const nav = [
  { to: "/demo", label: "Demo" },
  { to: "/how-it-works", label: "How it works" },
  { to: "/benchmarks", label: "Benchmarks" },
  { to: "/stack", label: "Stack" },
] as const

export const latency = [
  { k: "P50", v: "173ms" },
  { k: "P75", v: "189ms" },
  { k: "P90", v: "205ms" },
  { k: "P95", v: "214ms" },
]

export const budget = [
  { part: "Speech-to-text", ms: 38, pct: 22 },
  { part: "Hybrid retrieval", ms: 41, pct: 24 },
  { part: "Cross-encoder rerank", ms: 22, pct: 13 },
  { part: "Grounded generation", ms: 72, pct: 42 },
]

export const quality = [
  { k: "Groundedness", v: "94.2%" },
  { k: "Recall@5", v: "89.6%" },
  { k: "Word error rate", v: "7.8%" },
]

export const chunking = [
  {
    name: "Semantic",
    note: "Embedding-based splits that keep each chunk self-contained — the default lane for most questions.",
  },
  {
    name: "Fixed-overlap",
    note: "Sliding window with 15% overlap so answers that straddle a cut never fall through.",
  },
  {
    name: "Metadata-aware",
    note: "Section, heading and provenance preserved per chunk so citations stay grounded.",
  },
  {
    name: "Sentence-window",
    note: "The question's sentence plus its neighbours, kept together for precise, quote-ready spans.",
  },
]

export const stack = [
  "Sarvam STT",
  "bge-small-en-v1.5",
  "Qdrant HNSW + BM25",
  "Cross-encoder rerank",
  "Groq Llama-3.1-70B",
  "Cloudflare Workers",
  "MSMARCO-XI (8.8M passages)",
]
