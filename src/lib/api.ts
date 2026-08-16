// Set VITE_BACKEND_URL in .env.local for dev, or in Vercel env vars for production
// e.g. VITE_BACKEND_URL=https://YOUR-USERNAME-tilt-rag.hf.space
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export interface AskResponse {
  answer: string;
  status: string;
  transcript?: string;
  retrieved_context?: { matched_question?: string; query?: string; answer: string }[];
  total_latency_ms?: number;
  total_pipeline_latency_ms?: number;
  latency_ms?: number;
  stt_latency_ms?: number;
  retrieval_latency_ms?: number;
  rerank_latency_ms?: number;
  llm_latency_ms?: number;
}

/** Text query → RAG answer */
export async function askText(query: string): Promise<AskResponse> {
  const res = await fetch(`${BACKEND_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Backend error");
  }
  return res.json();
}

/** Audio Blob (WAV from MediaRecorder) → Sarvam STT → RAG answer */
export async function askVoice(audioBlob: Blob): Promise<AskResponse> {
  const form = new FormData();
  form.append("file", audioBlob, "recording.wav");
  const res = await fetch(`${BACKEND_URL}/voice-ask`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Voice backend error");
  }
  return res.json();
}
