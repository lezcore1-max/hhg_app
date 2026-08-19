import os
import sys
import time
import uuid
import asyncio
import threading
import numpy as np
import pandas as pd
import httpx
import requests
import pyarrow.dataset as ds
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv


# ── Windows UTF-8 console fix ──────────────────────────────────────────────
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()



from google import genai
from google.genai import types
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import faiss

# ── Configuration Constants ──────────────────────────────────────────────────
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "lezcore1-max/tilt-rag-data")
DATA_DIR = os.getenv("DATA_DIR", "/app/data" if os.path.isdir("/app/data") else ".")
os.makedirs(DATA_DIR, exist_ok=True)
print(f"[INIT] Using data directory: {DATA_DIR}", flush=True)

INDEX_Q_PATH = os.path.join(DATA_DIR, "small_hindi_passages.faiss")
PARQUET_PATH = os.path.join(DATA_DIR, "chunks_for_embedding.parquet")
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"  # 384-dim lightweight embedding model
VECTOR_DIM = 384  # multilingual-e5-small output dimension

# Devanagari script-aware tokenization regex (preserves Devanagari matras and viramas)
import re
HINDI_TOKEN_REGEX = re.compile(r'[^\s\.,;!?।॥\(\)\[\]\{\}"\':]+')

def tokenize_hindi(text: str) -> list[str]:
    """Preserves full Devanagari words without shattering vowels or viramas."""
    return HINDI_TOKEN_REGEX.findall(str(text).lower())


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables or .env file!", flush=True)
    gemini_client = None

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    print("⚠️ WARNING: SARVAM_API_KEY not found — /voice-ask and /ws/sarvam will fail until it is set.", flush=True)

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise Hindi RAG QA Engine",
    description="High-performance hybrid retrieval (FAISS + BM25) powered by intfloat/multilingual-e5-small and Gemini 3.1 Flash Lite",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Engine State ──────────────────────────────────────────────────────
corpus_chunk_ids = None
corpus_queries = None
corpus_answers = None
total_records = 0
mmap_vectors = None
index_q = None
bm25 = None
onnx_session = None
onnx_tokenizer = None
_engine_ready = False
_engine_error = None





# ── Data Download Utility ───────────────────────────────────────────────────
def _ensure_data_files():
    """Download FAISS index and Parquet from HuggingFace Hub with fast download."""
    from huggingface_hub import hf_hub_download

    # Clean up old large 2GB FAISS index if present to free disk space
    old_files = ["hindi_passages.faiss"]
    for old_f in old_files:
        old_path = os.path.join(DATA_DIR, old_f)
        if os.path.exists(old_path):
            print(f"🧹 Removing old file {old_f} to free disk space...", flush=True)
            try:
                os.remove(old_path)
                print(f"   ✅ Removed {old_f}", flush=True)
            except Exception as e:
                print(f"   ⚠️ Could not remove {old_f}: {e}", flush=True)

    expected_min_sizes = {
        "small_hindi_passages.faiss": 700 * 1024 * 1024,  # ~782.8 MB
        "chunks_for_embedding.parquet": 100 * 1024 * 1024  # ~149.5 MB
    }

    files_needed = ["small_hindi_passages.faiss", "chunks_for_embedding.parquet"]
    for fname in files_needed:
        fpath = os.path.join(DATA_DIR, fname)
        min_sz = expected_min_sizes.get(fname, 0)
        if not os.path.exists(fpath) or os.path.getsize(fpath) < min_sz:
            if os.path.exists(fpath):
                print(f"⚠️ Existing {fname} is incomplete ({os.path.getsize(fpath):,} bytes < {min_sz:,} bytes). Re-downloading...", flush=True)
                try:
                    os.remove(fpath)
                except Exception:
                    pass
            print(f"⬇️ Downloading {fname} from HuggingFace Hub ({HF_DATASET_REPO})...", flush=True)
            hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=fname,
                repo_type="dataset",
                local_dir=DATA_DIR,
                token=os.getenv("HF_TOKEN")
            )
            print(f"   ✅ {fname} ready ({os.path.getsize(fpath):,} bytes)!", flush=True)
        else:
            print(f"✅ {fname} exists ({os.path.getsize(fpath):,} bytes). Skipping download.", flush=True)


# ── Query Embedding (Lightweight ONNX multilingual-e5-small ~15MB) ─────────────
def get_query_embedding(query_text: str):
    """
    Generate 384-dim query embedding using ultra-lightweight ONNX Runtime (8-10ms, 15MB disk).
    No heavy PyTorch/CUDA installation required (fixes AWS disk quota).
    """
    global onnx_session, onnx_tokenizer
    input_text = f"query: {query_text}"

    # 1. Preferred: High-Performance ONNX Runtime (8-10 ms CPU)
    if onnx_session is not None and onnx_tokenizer is not None:
        try:
            enc = onnx_tokenizer.encode(input_text)
            input_ids = np.array([enc.ids], dtype=np.int64)
            attention_mask = np.array([enc.attention_mask], dtype=np.int64)
            
            input_names = [inp.name for inp in onnx_session.get_inputs()]
            feed = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = np.zeros_like(input_ids)
            
            outputs = onnx_session.run(None, feed)
            token_embeddings = outputs[0]  # [1, seq_len, 384]

            
            # Mean-pool over attention mask
            mask_exp = np.expand_dims(attention_mask, -1).astype(float)
            sum_emb = np.sum(token_embeddings * mask_exp, axis=1)
            sum_mask = np.clip(mask_exp.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = sum_emb / sum_mask
            
            # L2 normalize
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            norm_pooled = pooled / np.clip(norm, a_min=1e-12, a_max=None)
            return norm_pooled[0].astype(np.float32)
        except Exception as e:
            print(f"⚠️ ONNX encode error: {e}", flush=True)

    return None




# ── Background Engine Initialization ─────────────────────────────────────────
def _load_rag_engine_background():
    """Run all heavy data loading in a background thread so FastAPI starts instantly."""
    global corpus_chunk_ids, corpus_queries, corpus_answers, total_records, mmap_vectors, index_q, bm25, embed_model, _engine_ready, _engine_error
    try:
        print("=" * 60, flush=True)
        print("🚀 INITIALIZING HINDI RAG ENGINE (BACKGROUND THREAD)...", flush=True)
        print("=" * 60, flush=True)

        # 0. Auto-download data files from HuggingFace Hub if needed
        _ensure_data_files()

        # 1. Load Parquet columns into RAM (~180 MB RAM total, 0.01ms lookup)
        print(f"📦 Step 1/3: Loading in-memory corpus index from {PARQUET_PATH}...", flush=True)
        df_records = pd.read_parquet(PARQUET_PATH, columns=["chunk_id", "hindi_query", "hindi_answer"])
        corpus_chunk_ids = df_records["chunk_id"].tolist()
        corpus_queries = df_records["hindi_query"].tolist()
        corpus_answers = df_records["hindi_answer"].tolist()
        total_records = len(corpus_queries)
        del df_records  # Free DataFrame memory overhead
        print(f"   ✅ Loaded {total_records:,} corpus entries into fast RAM lists (0.01ms lookup)!", flush=True)

        # 2. Memory-Map 384-dim Vector Matrix (0 MB RAM overhead!)
        print(f"⚡ Step 2/3: Memory-mapping vector matrix from {INDEX_Q_PATH} (dim={VECTOR_DIM})...", flush=True)
        try:
            mmap_vectors = np.memmap(INDEX_Q_PATH, dtype="float32", mode="r", offset=45, shape=(total_records, VECTOR_DIM))
            print(f"   ✅ Vector matrix memory-mapped ({total_records:,} vectors × {VECTOR_DIM}-dim, 0 MB RAM)!", flush=True)
        except Exception as mmap_err:
            print(f"⚠️ Vector mmap fallback ({mmap_err}); loading FAISS index...", flush=True)
            index_q = faiss.read_index(INDEX_Q_PATH)
            print(f"   ✅ FAISS index loaded with {index_q.ntotal:,} vectors!", flush=True)

        # 3. Build BM25 Index with script-aware Devanagari tokenization
        print("🔍 Step 3/3: Building Devanagari script-aware BM25 keyword index...", flush=True)
        tokenized_queries = [tokenize_hindi(q) for q in corpus_queries]
        bm25 = BM25Okapi(tokenized_queries)

        # 4. Load & pre-warm ONNX Embedder (~15MB disk, 8-10ms latency)
        print(f"🧠 Loading ONNX {EMBED_MODEL_NAME} runtime...", flush=True)
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            
            onnx_model_path = hf_hub_download(EMBED_MODEL_NAME, "onnx/model.onnx", local_dir=DATA_DIR)
            onnx_tok_path = hf_hub_download(EMBED_MODEL_NAME, "tokenizer.json", local_dir=DATA_DIR)
            
            onnx_tokenizer = Tokenizer.from_file(onnx_tok_path)
            onnx_tokenizer.enable_truncation(max_length=512)
            
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 4
            onnx_session = ort.InferenceSession(onnx_model_path, sess_opts, providers=["CPUExecutionProvider"])
            
            # Warm up
            _ = get_query_embedding("warm up")
            print(f"   ✅ ONNX {EMBED_MODEL_NAME} loaded and warmed up (~9ms CPU latency, 0 PyTorch RAM)!", flush=True)
        except Exception as onnx_err:
            print(f"   ⚠️ ONNX load note: {onnx_err}", flush=True)


        _engine_ready = True

        print("=" * 60, flush=True)
        print(f"🎯 HHG RAG ENGINE READY — {total_records:,} chunks indexed ({EMBED_MODEL_NAME})!", flush=True)
        print("=" * 60, flush=True)


    except Exception as e:
        _engine_error = str(e)
        print(f"🚨 ENGINE LOAD FAILED: {e}", flush=True)


@app.on_event("startup")
def startup_event():
    """Start FastAPI immediately, load data in background thread."""
    print("🟢 FastAPI started — launching RAG engine loader in background thread...", flush=True)
    t = threading.Thread(target=_load_rag_engine_background, daemon=True)
    t.start()


# ── Request / Guardrail Schemas ──────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    k: int = 5


def guardrail_check(top_results: list, top_sim_score: float = 0.0,
                     rrf_threshold: float = 0.01, semantic_sim_threshold: float = 0.45):
    if not top_results:
        return {"should_answer": False, "reason": "no_results", "top_score": 0.0, "semantic_sim": 0.0}

    top = top_results[0]
    score_val = float(top.get("score", 0.0))
    semantic_sim = float(top_sim_score) if top_sim_score > 0 else score_val

    if score_val < rrf_threshold or semantic_sim < semantic_sim_threshold:
        return {
            "should_answer": False,
            "reason": "low_confidence_off_topic" if score_val < rrf_threshold else "semantic_mismatch",
            "top_score": round(score_val, 3),
            "semantic_sim": round(semantic_sim, 3),
        }

    return {
        "should_answer": True,
        "reason": "grounded",
        "top_score": round(score_val, 3),
        "semantic_sim": round(semantic_sim, 3),
        "grounded_answer": top["answer"],
        "retrieved_context": top_results,
    }



# ── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
def health_check():
    """Always returns 200 OK immediately — cloud container healthcheck safe."""
    if _engine_ready:
        return {
            "status": "ready",
            "engine": "Hindi RAG QA Engine v2.2",
            "records_indexed": total_records,
            "models": {
                "embedding": EMBED_MODEL_NAME,
                "dimension": VECTOR_DIM,
                "llm": "gemini-3.1-flash-lite"
            }
        }
    elif _engine_error:
        return {"status": "error", "detail": _engine_error}
    else:
        return {"status": "loading", "message": "RAG engine is initializing in background, please wait..."}


@app.post("/ask")
async def ask_question(req: QueryRequest):
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not _engine_ready:
        msg = f"Engine loading error: {_engine_error}" if _engine_error else "RAG engine is still loading, please retry in a moment."
        raise HTTPException(status_code=503, detail=msg)

    if (mmap_vectors is None and index_q is None) or corpus_queries is None:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized.")

    t0_rag = time.perf_counter()

    try:
        # 1. High-Performance Candidate Vector Search (50 candidates in ~5ms, 0 MB RAM)
        candidate_k = 50
        dense_idx = []
        dense_scores = []
        dense_score_map = {}
        q_emb = get_query_embedding(query_text)

        try:
            if q_emb is not None:
                if mmap_vectors is not None:
                    # 0 MB RAM vector dot product with fast argpartition
                    scores = np.dot(mmap_vectors, q_emb)
                    part_idx = np.argpartition(scores, -candidate_k)[-candidate_k:]
                    dense_idx = part_idx[np.argsort(scores[part_idx])[::-1]]
                    dense_scores = scores[dense_idx]
                    for d_i, d_sc in zip(dense_idx, dense_scores):
                        if 0 <= d_i < total_records:
                            dense_score_map[int(d_i)] = float(d_sc)
                elif index_q is not None:
                    q_emb_arr = np.array([q_emb], dtype="float32")
                    dense_scores_arr, dense_idx_arr = index_q.search(q_emb_arr, candidate_k)
                    dense_idx = dense_idx_arr[0]
                    dense_scores = dense_scores_arr[0]
                    for d_i, d_sc in zip(dense_idx, dense_scores):
                        if 0 <= d_i < total_records:
                            dense_score_map[int(d_i)] = float(d_sc)
        except Exception as embed_err:
            print(f"⚠️ Dense search error: {embed_err}", flush=True)


        # 2. Fast Candidate BM25 Scoring with Devanagari Tokenizer
        cand_doc_ids = list(dense_score_map.keys())
        bm25_score_map = {}
        if cand_doc_ids:
            q_tokens = tokenize_hindi(query_text)
            bm25_batch = bm25.get_batch_scores(q_tokens, cand_doc_ids)
            for d_id, b_sc in zip(cand_doc_ids, bm25_batch):
                bm25_score_map[d_id] = float(b_sc)


        # 3. Reciprocal Rank Fusion (RRF) over Candidate Pool
        k_rrf = 60
        rrf_scores = {}
        for rank, d_id in enumerate(dense_idx):
            d_id_int = int(d_id)
            if 0 <= d_id_int < total_records:
                rrf_scores[d_id_int] = rrf_scores.get(d_id_int, 0.0) + (1.0 / (k_rrf + rank + 1))

        bm25_sorted_cand = sorted(bm25_score_map.keys(), key=lambda x: bm25_score_map[x], reverse=True)
        for rank, d_id in enumerate(bm25_sorted_cand):
            rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (1.0 / (k_rrf + rank + 1))

        sorted_rrf_pairs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:max(req.k, 4)]
        top_candidate_indices = [pair[0] for pair in sorted_rrf_pairs]

        # 4. Instant In-Memory Row Assembly (0.01 ms!)
        max_bm25 = max(bm25_score_map.values()) if bm25_score_map and max(bm25_score_map.values()) > 0 else 1.0

        retrieved_docs = []
        for rank, idx in enumerate(top_candidate_indices):
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= total_records:
                continue

            if idx_int in dense_score_map:
                score_val = float(dense_score_map[idx_int])
            else:
                bm25_norm = float(bm25_score_map.get(idx_int, 0.0)) / max_bm25 if max_bm25 > 0 else 0.5
                score_val = 0.52 + (bm25_norm * 0.33)

            cid = corpus_chunk_ids[idx_int]
            hq = corpus_queries[idx_int]
            ha = corpus_answers[idx_int]

            retrieved_docs.append({
                "chunk_id": str(cid),
                "passage_id": str(cid.split('#')[0] if '#' in str(cid) else cid),
                "query_id": str(cid.split('#')[0] if '#' in str(cid) else cid),
                "strategy": "metadata_aware",
                "score": round(score_val, 3),
                "query": hq,
                "matched_question": hq,
                "answer": ha,
                "chunk_text": ha,
            })
        retrieved_docs = retrieved_docs[:max(req.k, 4)]


        retrieval_latency = (time.perf_counter() - t0_rag) * 1000

        # 5. Guardrail Check (instant 0.001ms check with already-computed similarity)
        top_sim = dense_score_map.get(top_candidate_indices[0], 0.85) if top_candidate_indices else 0.0
        check = guardrail_check(retrieved_docs, top_sim_score=top_sim)


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

        # 6. Build Compact Grounded Prompt for Gemini LLM
        context_docs = "\n\n".join(
            f"Q: {doc['query']}\nA: {doc['answer']}"
            for doc in check["retrieved_context"][:3]
        )

        prompt = f"""Context:
{context_docs}

User Question: {query_text}
Answer:"""

        # 7. Call Gemini 3.1 Flash Lite for low-latency grounded generation
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


# ── WebSocket Proxy for Sarvam Realtime STT ──────────────────────────────────
SARVAM_WS_BASE = "wss://api.sarvam.ai/speech-to-text-realtime/ws"


@app.websocket("/ws/sarvam")
async def sarvam_ws_proxy(client_ws: WebSocket):
    """
    Proxies browser WebSocket to Sarvam's realtime streaming endpoint,
    injecting the Api-Subscription-Key header that browsers cannot send.
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
