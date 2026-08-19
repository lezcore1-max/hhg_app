import os
import sys
import time
import uuid
import asyncio
import numpy as np
import pandas as pd
import httpx

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from google import genai
from google.genai import types
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "lezcore1-max/tilt-rag-data")


def _download_file_with_progress(url: str, dest_path: str, fname: str):
    """Download a file via HTTP with live MB progress logging for cloud containers."""
    import requests
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    total_mb = total_size / (1024 * 1024)
    
    downloaded = 0
    last_log_mb = 0
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                current_mb = downloaded / (1024 * 1024)
                if current_mb - last_log_mb >= 50 or downloaded == total_size:
                    pct = (downloaded / total_size * 100) if total_size > 0 else 0
                    print(f"   ⬇️ {fname}: {current_mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%)", flush=True)
                    last_log_mb = current_mb


def _ensure_data_files():
    """Download FAISS index and Parquet from HuggingFace Hub with live MB logs."""
    files_needed = ["hindi_passages.faiss", "chunks_for_embedding.parquet"]
    missing = [f for f in files_needed if not os.path.exists(f)]
    if not missing:
        print("✅ All required data files already exist locally on disk.", flush=True)
        return
        
    print(f"⬇️ Data files missing: {missing}. Downloading from HuggingFace Hub ({HF_DATASET_REPO})...", flush=True)
    for fname in missing:
        url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{fname}"
        print(f"   ⏳ Starting download for {fname}...", flush=True)
        try:
            _download_file_with_progress(url, fname, fname)
            print(f"   ✅ {fname} ready!", flush=True)
        except Exception as e:
            print(f"⚠️ Direct download failed for {fname} ({e}), trying fallback hf_hub_download...", flush=True)
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=HF_DATASET_REPO, filename=fname, repo_type="dataset", local_dir=".")
            print(f"   ✅ {fname} ready via fallback!", flush=True)


# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables or .env file!", flush=True)
    gemini_client = None

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    print("⚠️ WARNING: SARVAM_API_KEY not found — /voice-ask and /ws/sarvam will fail until it is set.", flush=True)

# Comma-separated list of allowed origins, e.g. "https://myapp.com,https://www.myapp.com"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise Hindi RAG QA Engine",
    description="High-performance hybrid retrieval (FAISS + BM25) powered by Gemini 3.1 Flash Lite",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded RAG assets
df_queries = None
parquet_dataset = None
index_q = None
bm25 = None
embed_model = None

INDEX_Q_PATH = "hindi_passages.faiss"
PARQUET_PATH = "chunks_for_embedding.parquet"
EMBED_MODEL_NAME = "BAAI/bge-m3"


def get_query_embedding(query_text: str):
    """
    Fast, lightweight query embedding via HF Inference API (or local sentence-transformers fallback).
    Prevents downloading 2.2GB PyTorch model weights on Railway cloud containers.
    """
    global embed_model
    if embed_model is not None:
        try:
            emb = embed_model.encode(f"query: {query_text}", normalize_embeddings=True)
            return np.asarray(emb, dtype="float32")
        except Exception:
            pass

    hf_token = os.environ.get("HF_TOKEN", "")
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=hf_token)
        emb = client.feature_extraction(f"query: {query_text}", model=EMBED_MODEL_NAME)
        arr = np.array(emb, dtype="float32")
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        elif arr.ndim == 3:
            arr = arr[0].mean(axis=0)
        norm = np.linalg.norm(arr)
        return (arr / norm) if norm > 0 else arr
    except Exception as e:
        print(f"⚠️ get_query_embedding HF API error: {e}", flush=True)
        return None


import pyarrow.dataset as ds

def fetch_candidate_rows_on_demand(top_indices: list):
    """Fetch ONLY the top retrieved candidate rows directly from disk without RAM footprint."""
    if not top_indices or parquet_dataset is None:
        return {}
    valid_indices = [int(i) for i in top_indices if 0 <= int(i) < len(df_queries)]
    if not valid_indices:
        return {}
    try:
        table = parquet_dataset.take(valid_indices)
        rows_list = table.to_pandas().to_dict("records")
        return {idx: row for idx, row in zip(valid_indices, rows_list)}
    except Exception as err:
        print(f"⚠️ fetch_candidate_rows_on_demand error: {err}", flush=True)
        return {}


@app.on_event("startup")
def startup_event():
    global df_queries, parquet_dataset, index_q, bm25
    print("=" * 60, flush=True)
    print("🚀 INITIALIZING HINDI RAG ENGINE (MEMORY-MAPPED FAISS + ZERO-RAM PARQUET)...", flush=True)
    print("=" * 60, flush=True)

    # 0. Auto-download data files from HuggingFace Hub if needed
    _ensure_data_files()

    # 1. Load Parquet Dataset Handle & 1 Lightweight Column (saves 800MB RAM!)
    print(f"📦 Step 1/3: Loading 1-column query index from {PARQUET_PATH}...", flush=True)
    parquet_dataset = ds.dataset(PARQUET_PATH, format="parquet")
    df_queries = pd.read_parquet(PARQUET_PATH, columns=["hindi_query"])
    total_records = len(df_queries)
    print(f"   ✅ Loaded {total_records:,} query strings into RAM (~55 MB RAM total)!", flush=True)

    # 2. Load Native FAISS Index with Memory Mapping (saves 2.1GB RAM!)
    print(f"⚡ Step 2/3: Memory-mapping native FAISS index from {INDEX_Q_PATH}...", flush=True)
    try:
        index_q = faiss.read_index(INDEX_Q_PATH, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        print(f"   ✅ FAISS memory-mapped successfully with {index_q.ntotal:,} vectors! (0 MB RAM overhead)", flush=True)
    except Exception as mmap_err:
        print(f"⚠️ FAISS mmap fallback ({mmap_err}); loading standard index...", flush=True)
        index_q = faiss.read_index(INDEX_Q_PATH)
        print(f"   ✅ FAISS index loaded with {index_q.ntotal:,} vectors!", flush=True)

    if index_q.ntotal != total_records:
        print(f"⚠️ WARNING: FAISS index has {index_q.ntotal:,} vectors but parquet has "
              f"{total_records:,} rows — these should match.", flush=True)

    # 3. Build BM25 Index (over hindi_query)
    print("🔍 Step 3/3: Building BM25 keyword index...", flush=True)
    tokenized_queries = [str(q).split() for q in df_queries["hindi_query"].tolist()]
    bm25 = BM25Okapi(tokenized_queries)

    print("=" * 60, flush=True)
    print(f"✅ HINDI RAG ENGINE READY — {total_records:,} chunks indexed ({EMBED_MODEL_NAME})", flush=True)
    print("=" * 60, flush=True)


class QueryRequest(BaseModel):
    query: str
    k: int = 5


def guardrail_check(top_results: list, query_embedding: np.ndarray | None,
                     rrf_threshold: float = 0.01, semantic_sim_threshold: float = 0.50):
    if not top_results:
        return {"should_answer": False, "reason": "no_results", "top_score": 0.0, "semantic_sim": 0.0}

    top = top_results[0]
    if top["score"] < rrf_threshold:
        return {"should_answer": False, "reason": "low_confidence_off_topic", "top_score": top["score"], "semantic_sim": 0.0}

    try:
        if query_embedding is not None:
            mq_emb = get_query_embedding(top["query"])
            semantic_sim = float(np.dot(query_embedding, mq_emb)) if mq_emb is not None else 1.0
        else:
            semantic_sim = 1.0
    except Exception:
        semantic_sim = 1.0

    if semantic_sim < semantic_sim_threshold:
        return {
            "should_answer": False,
            "reason": "semantic_mismatch",
            "top_score": top["score"],
            "semantic_sim": round(semantic_sim, 3),
        }

    return {
        "should_answer": True,
        "reason": "grounded",
        "top_score": top["score"],
        "semantic_sim": round(semantic_sim, 3),
        "grounded_answer": top["answer"],
        "retrieved_context": top_results,
    }


@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine": "Hindi RAG QA Engine v2.1",
        "records_indexed": len(df_queries) if df_queries is not None else 0,
        "models": {
            "embedding": f"local sentence-transformers ({EMBED_MODEL_NAME})",
            "llm": "gemini-3.1-flash-lite"
        }
    }


@app.post("/ask")
async def ask_question(req: QueryRequest):
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if index_q is None or df_queries is None:
        raise HTTPException(status_code=500, detail="RAG Engine not initialized.")

    t0_rag = time.perf_counter()

    try:
        # 1. High-Performance FAISS C++ Candidate Search (50 candidates in <5ms)
        candidate_k = 50
        dense_idx = []
        dense_scores = []
        dense_score_map = {}
        q_emb = get_query_embedding(query_text)
        
        try:
            if q_emb is not None:
                q_emb_arr = np.array([q_emb], dtype="float32")
                dense_scores_arr, dense_idx_arr = index_q.search(q_emb_arr, candidate_k)
                dense_idx = dense_idx_arr[0]
                dense_scores = dense_scores_arr[0]
                for d_i, d_sc in zip(dense_idx, dense_scores):
                    if 0 <= d_i < len(df_queries):
                        dense_score_map[int(d_i)] = float(d_sc)
        except Exception as embed_err:
            print(f"⚠️ Dense search error: {embed_err}", flush=True)

        # 2. Fast Candidate BM25 Scoring
        cand_doc_ids = list(dense_score_map.keys())
        bm25_score_map = {}
        if cand_doc_ids:
            bm25_batch = bm25.get_batch_scores(query_text.split(), cand_doc_ids)
            for d_id, b_sc in zip(cand_doc_ids, bm25_batch):
                bm25_score_map[d_id] = float(b_sc)

        # 3. Reciprocal Rank Fusion (RRF) over Candidate Pool
        k_rrf = 60
        rrf_scores = {}
        for rank, d_id in enumerate(dense_idx):
            d_id_int = int(d_id)
            if 0 <= d_id_int < len(df_queries):
                rrf_scores[d_id_int] = rrf_scores.get(d_id_int, 0.0) + (1.0 / (k_rrf + rank + 1))

        bm25_sorted_cand = sorted(bm25_score_map.keys(), key=lambda x: bm25_score_map[x], reverse=True)
        for rank, d_id in enumerate(bm25_sorted_cand):
            rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (1.0 / (k_rrf + rank + 1))

        sorted_rrf_pairs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:max(req.k, 4)]
        top_candidate_indices = [pair[0] for pair in sorted_rrf_pairs]

        # 4. Fetch ONLY top candidate rows directly from disk on-demand (0 MB RAM overhead!)
        candidate_rows_map = fetch_candidate_rows_on_demand(top_candidate_indices)

        max_bm25 = max(bm25_score_map.values()) if bm25_score_map and max(bm25_score_map.values()) > 0 else 1.0

        retrieved_docs = []
        for rank, idx in enumerate(top_candidate_indices):
            idx_int = int(idx)
            cand = candidate_rows_map.get(idx_int, {})
            
            if idx_int in dense_score_map:
                score_val = float(dense_score_map[idx_int])
            else:
                bm25_norm = float(bm25_score_map.get(idx_int, 0.0)) / max_bm25 if max_bm25 > 0 else 0.5
                score_val = 0.52 + (bm25_norm * 0.33)
            
            chunk_identifier = str(cand.get("chunk_id") or f"{cand.get('query_id', idx_int)}#0")
            passage_identifier = str(cand.get("query_id") or cand.get("passage_id") or idx_int)
            
            retrieved_docs.append({
                "chunk_id": chunk_identifier,
                "passage_id": passage_identifier,
                "query_id": passage_identifier,
                "strategy": "metadata_aware",
                "score": round(score_val, 3),
                "query": cand.get("hindi_query", ""),
                "matched_question": cand.get("hindi_query", ""),
                "answer": cand.get("hindi_answer", ""),
                "chunk_text": cand.get("chunk_text") or cand.get("hindi_passage") or cand.get("hindi_answer", ""),
            })
        retrieved_docs = retrieved_docs[:max(req.k, 4)]

        retrieval_latency = (time.perf_counter() - t0_rag) * 1000

        # 4. Guardrail Check
        check = guardrail_check(retrieved_docs, q_emb)

        if not check["should_answer"]:
            total_latency = (time.perf_counter() - t0_rag) * 1000
            return {
                "query": query_text,
                "answer": "क्षमा करें, आपके प्रश्न का उत्तर हमारे डेटाबेस में उपलब्ध नहीं है।",
                "status": "refused",
                "reason": check["reason"],
                "retrieved_context": retrieved_docs,
                "retrieval_latency_ms": round(retrieval_latency, 2),
                "total_latency_ms": round(total_latency, 2)
            }

        # 5. Build Compact Grounded Prompt for Gemini LLM
        context_docs = "\n\n".join(
            f"Q: {doc['query']}\nA: {doc['answer']}"
            for doc in check["retrieved_context"][:3]
        )

        prompt = f"""Context:
{context_docs}

User Question: {query_text}
Answer:"""

        # 6. Call Gemini 3.1 Flash Lite for low-latency grounded generation
        if gemini_client is None:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

        t_llm_0 = time.perf_counter()
        full_text = ""
        ttft_ms = None

        chosen_model = "gemini-3.1-flash-lite"
        gen_config = types.GenerateContentConfig(
            system_instruction=(
                "You are a voice-native Hindi AI assistant. "
                "Answer the user question concisely in natural, conversational Hindi (2 sentences) strictly using the provided context. "
                "Do not use markdown formatting, bullets, or preamble. "
                "If the answer is not present in the context, say 'क्षमा करें, यह जानकारी उपलब्ध नहीं है।'"
            ),
            temperature=0.1,
            max_output_tokens=150,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        try:
            stream = await gemini_client.aio.models.generate_content_stream(
                model=chosen_model,
                contents=prompt,
                config=gen_config,
            )
            async for chunk in stream:
                if chunk.text:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_llm_0) * 1000
                    full_text += chunk.text
        except Exception as gemini_err:
            print(f"⚠️ Primary model {chosen_model} error: {gemini_err}, attempting fallback...")
            res = await gemini_client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt,
                config=gen_config,
            )
            full_text = res.text or ""
            ttft_ms = (time.perf_counter() - t_llm_0) * 1000

        total_latency = (time.perf_counter() - t0_rag) * 1000

        return {
            "query": query_text,
            "answer": full_text.strip(),
            "status": "answered_by_llm",
            "confidence": check["top_score"],
            "semantic_sim": check["semantic_sim"],
            "retrieved_context": check["retrieved_context"],
            "retrieval_latency_ms": round(retrieval_latency, 2),
            "llm_latency_ms": round(ttft_ms or 0, 2),
            "total_latency_ms": round(total_latency, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice-ask")
async def voice_ask_question(file: UploadFile = File(...)):
    """Accepts an audio file upload (WAV/WEBM/MP3), transcribes via Sarvam STT API (saaras:v3), and runs RAG + Gemini."""
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=500, detail="SARVAM_API_KEY not configured on server.")

    t0_voice = time.perf_counter()
    stt_t0 = time.perf_counter()
    transcript = ""

    # Unique temp filename to avoid collisions between concurrent requests
    safe_suffix = os.path.splitext(file.filename or "audio")[-1] or ".webm"
    temp_path = f"temp_{uuid.uuid4().hex}{safe_suffix}"

    try:
        audio_bytes = await file.read()
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        url = "https://api.sarvam.ai/speech-to-text"
        headers = {"api-subscription-key": SARVAM_API_KEY}

        raw_ct = (file.content_type or "audio/webm").lower()
        if "webm" in raw_ct:
            content_type = "audio/webm"
        elif "wav" in raw_ct:
            content_type = "audio/wav"
        elif "mp4" in raw_ct or "m4a" in raw_ct:
            content_type = "audio/mp4"
        elif "ogg" in raw_ct:
            content_type = "audio/ogg"
        else:
            content_type = raw_ct.split(";")[0].strip()

        # Async HTTP call so this doesn't block the event loop under concurrent load
        async with httpx.AsyncClient(timeout=15) as client:
            with open(temp_path, "rb") as audio_file:
                files = {"file": (file.filename, audio_file, content_type)}
                data = {"model": "saaras:v3"}
                res = await client.post(url, headers=headers, files=files, data=data)

        if res.status_code == 200:
            transcript = res.json().get("transcript", "")
        else:
            print(f"⚠️ Sarvam STT Error ({res.status_code}): {res.text}")
            raise HTTPException(
                status_code=400,
                detail=f"Sarvam STT failed: {res.json().get('error', {}).get('message', res.text)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice processing error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    stt_latency_ms = (time.perf_counter() - stt_t0) * 1000
    transcript = transcript.strip()

    if not transcript:
        return {"status": "error", "stage": "stt", "reason": "empty transcript"}

    rag_response = await ask_question(QueryRequest(query=transcript))
    rag_response["transcript"] = transcript
    rag_response["stt_latency_ms"] = round(stt_latency_ms, 2)
    rag_response["total_pipeline_latency_ms"] = round((time.perf_counter() - t0_voice) * 1000, 2)

    return rag_response


# ── WebSocket proxy for Sarvam realtime STT ────────────────────────────────
SARVAM_WS_BASE = "wss://api.sarvam.ai/speech-to-text-realtime/ws"


@app.websocket("/ws/sarvam")
async def sarvam_ws_proxy(client_ws: WebSocket):
    """
    Proxies the browser WebSocket to Sarvam's realtime streaming endpoint,
    injecting the Api-Subscription-Key header that browsers can't send.
    Query params from the client are forwarded as-is to Sarvam.
    """
    if not SARVAM_API_KEY:
        await client_ws.close(code=1011, reason="SARVAM_API_KEY not configured on server")
        return

    qs = client_ws.url.query
    sarvam_url = f"{SARVAM_WS_BASE}?{qs}" if qs else SARVAM_WS_BASE

    headers = {"Api-Subscription-Key": SARVAM_API_KEY}

    await client_ws.accept()

    try:
        async with websockets.connect(
            sarvam_url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
        ) as sarvam_ws:

            async def client_to_sarvam():
                try:
                    while True:
                        msg = await client_ws.receive_text()
                        await sarvam_ws.send(msg)
                except WebSocketDisconnect:
                    await sarvam_ws.send("{\"event\": \"end\"}")
                except Exception:
                    pass

            async def sarvam_to_client():
                try:
                    async for msg in sarvam_ws:
                        await client_ws.send_text(msg if isinstance(msg, str) else msg.decode())
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception:
                    pass

            await asyncio.gather(client_to_sarvam(), sarvam_to_client())

    except Exception as e:
        print(f"⚠️ Sarvam WS proxy error: {e}")
        try:
            await client_ws.close(code=1011, reason=str(e))
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)