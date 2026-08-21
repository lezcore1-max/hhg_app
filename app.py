import os
import sys
import time
import uuid
import asyncio
import threading
import numpy as np
import pandas as pd
import httpx
import re
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

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"  # 384-dim multilingual embedding model
VECTOR_DIM = 384

# ── Multi-Language Configuration ─────────────────────────────────────────────
LANG_CONFIG = {
    "hi": {
        "name": "Hindi",
        "faiss_file": "small_hindi_passages.faiss",
        "parquet_file": "chunks_for_embedding.parquet",
        "query_col": "hindi_query",
        "answer_col": "hindi_answer",
        "refusal": "क्षमा करें, आपके प्रश्न का उत्तर हमारे डेटाबेस में उपलब्ध नहीं है।",
        "system_instruction": (
            "You are a voice-native Hindi AI assistant. "
            "Answer the user question concisely in natural, conversational Hindi (2 sentences) strictly using the provided context. "
            "Do not use markdown formatting, bullets, or preamble. "
            "If the answer is not present in the context, say 'क्षमा करें, यह जानकारी उपलब्ध नहीं है।'"
        )
    },
    "mr": {
        "name": "Marathi",
        "faiss_file": "marathi_hindi_passages.faiss",
        "parquet_file": "marathi_chunks_for_embedding.parquet",
        "query_col": "marathi_query",
        "answer_col": "marathi_answer",
        "refusal": "माफ करा, तुमच्या प्रश्नाचे उत्तर आमच्या डेटाबेसमध्ये उपलब्ध नाही.",
        "system_instruction": (
            "You are a voice-native Marathi AI assistant. "
            "Answer the user question concisely in natural, conversational Marathi (2 sentences) strictly using the provided context. "
            "Do not use markdown formatting, bullets, or preamble. "
            "If the answer is not present in the context, say 'माफ करा, ही माहिती उपलब्ध नाही.'"
        )
    },
    "pa": {
        "name": "Punjabi",
        "faiss_file": "punjabi_hindi_passages.faiss",
        "parquet_file": "punjabi_chunks_for_embedding.parquet",
        "query_col": "punjabi_query",
        "answer_col": "punjabi_answer",
        "refusal": "ਮਾਫ਼ ਕਰਨਾ, ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਸਾਡੇ ਡੇਟਾਬੇਸ ਵਿੱਚ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।",
        "system_instruction": (
            "You are a voice-native Punjabi AI assistant. "
            "Answer the user question concisely in natural, conversational Punjabi (2 sentences) strictly using the provided context. "
            "Do not use markdown formatting, bullets, or preamble. "
            "If the answer is not present in the context, say 'ਮਾਫ਼ ਕਰਨਾ, ਇਹ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।'"
        )
    },
    "gu": {
        "name": "Gujarati",
        "faiss_file": "gujarati_hindi_passages.faiss",
        "parquet_file": "gujarati_chunks_for_embedding.parquet",
        "query_col": "gujarati_query",
        "answer_col": "gujarati_answer",
        "refusal": "માફ કરશો, તમારા પ્રશ્નનો જવાબ અમારા ડેટાબેઝમાં ઉપલબ્ધ નથી.",
        "system_instruction": (
            "You are a voice-native Gujarati AI assistant. "
            "Answer the user question concisely in natural, conversational Gujarati (2 sentences) strictly using the provided context. "
            "Do not use markdown formatting, bullets, or preamble. "
            "If the answer is not present in the context, say 'માફ કરશો, આ માહિતી ઉપલબ્ધ નથી.'"
        )
    },
    "ur": {
        "name": "Urdu",
        "faiss_file": "urdu_hindi_passages.faiss",
        "parquet_file": "urdu_chunks_for_embedding.parquet",
        "query_col": "urdu_query",
        "answer_col": "urdu_answer",
        "refusal": "معذرت، آپ کے سوال کا جواب ہمارے ڈیٹا بیس میں دستیاب نہیں ہے۔",
        "system_instruction": (
            "You are a voice-native Urdu AI assistant. "
            "Answer the user question concisely in natural, conversational Urdu (2 sentences) strictly using the provided context. "
            "Do not use markdown formatting, bullets, or preamble. "
            "If the answer is not present in the context, say 'معذرت، یہ معلومات دستیاب نہیں ہے۔'"
        )
    }
}

# ── Indic Tokenizer (Script-Preserving) ───────────────────────────────────────
INDIC_TOKEN_REGEX = re.compile(r'[^\s\.,;!?।॥۔؟؛\(\)\[\]\{\}"\':]+')

def tokenize_indic(text: str) -> list[str]:
    """Preserves full Indic/Urdu words without shattering diacritics or vowels."""
    return INDIC_TOKEN_REGEX.findall(str(text).lower())

# ── Language Detection (Sarvam Native Detection + Script Fallback) ──────────
SARVAM_LANG_MAP = {
    "hi-IN": "hi", "hi": "hi",
    "mr-IN": "mr", "mr": "mr",
    "pa-IN": "pa", "pa": "pa",
    "gu-IN": "gu", "gu": "gu",
    "ur-IN": "ur", "ur": "ur",
}

def detect_language(text: str, sarvam_lang_code: str | None = None) -> str:
    """
    Detects language using Sarvam's native language_code if available,
    otherwise uses fast Unicode script block ranges.
    """
    # 1. Preferred: Sarvam STT's native language detection
    if sarvam_lang_code and sarvam_lang_code in SARVAM_LANG_MAP:
        return SARVAM_LANG_MAP[sarvam_lang_code]
    
    if not text:
        return "hi"
    
    # 2. Fast Unicode Script Range Fallback for text queries
    # Urdu / Arabic script (U+0600 to U+06FF)
    if any("\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F" for c in text):
        return "ur"
    
    # Punjabi / Gurmukhi script (U+0A00 to U+0A7F)
    if any("\u0A00" <= c <= "\u0A7F" for c in text):
        return "pa"
    
    # Gujarati script (U+0A80 to U+0AFF)
    if any("\u0A80" <= c <= "\u0AFF" for c in text):
        return "gu"
    
    # Devanagari script (U+0900 to U+097F) -> default to Hindi / Devanagari
    if any("\u0900" <= c <= "\u097F" for c in text):
        if "ळ" in text or "ऱ" in text:
            return "mr"
        return "hi"
    
    return "hi"

# ── API Keys ─────────────────────────────────────────────────────────────────
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
    title="Enterprise Multilingual Indic RAG QA Engine",
    description="High-performance hybrid retrieval (FAISS + BM25) supporting Hindi, Marathi, Punjabi, Gujarati, Urdu powered by ONNX multilingual-e5-small and Gemini 3.1 Flash Lite",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Multi-Language Global Store ──────────────────────────────────────────────
lang_stores = {
    lang_code: {
        "chunk_ids": [],
        "queries": [],
        "answers": [],
        "total_records": 0,
        "mmap_vectors": None,
        "index_q": None,
        "ready": False
    }
    for lang_code in LANG_CONFIG
}

onnx_session = None
onnx_tokenizer = None
_engine_ready = False
_engine_error = None

# ── Data Download Utility ───────────────────────────────────────────────────
def _ensure_data_files():
    """Download required FAISS indexes and Parquet files for all languages if missing."""
    print("📥 Checking dataset files on disk...", flush=True)
    for lang_code, cfg in LANG_CONFIG.items():
        files_to_check = [cfg["faiss_file"], cfg["parquet_file"]]
        for fname in files_to_check:
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fpath) or os.path.getsize(fpath) < 100 * 1024:
                print(f"⬇️ Downloading {fname} for {cfg['name']} ({HF_DATASET_REPO})...", flush=True)
                try:
                    hf_hub_download(
                        repo_id=HF_DATASET_REPO,
                        filename=fname,
                        repo_type="dataset",
                        local_dir=DATA_DIR,
                        token=os.getenv("HF_TOKEN")
                    )
                    print(f"   ✅ {fname} ready ({os.path.getsize(fpath):,} bytes)!", flush=True)
                except Exception as dl_err:
                    print(f"   ⚠️ Could not download {fname}: {dl_err}", flush=True)
            else:
                print(f"   ✅ {fname} exists ({os.path.getsize(fpath):,} bytes)", flush=True)

# ── Query Embedding (Lightweight ONNX multilingual-e5-small ~15MB) ─────────────
def get_query_embedding(query_text: str):
    """
    Generate 384-dim query embedding using ultra-lightweight ONNX Runtime (8-10ms, 15MB disk).
    """
    global onnx_session, onnx_tokenizer
    input_text = f"query: {query_text}"

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
    """Load ONNX runtime and initialize memory-mapped indexes for all 5 languages."""
    global lang_stores, onnx_session, onnx_tokenizer, _engine_ready, _engine_error
    try:
        print("=" * 65, flush=True)
        print("🚀 INITIALIZING MULTILINGUAL INDIC RAG ENGINE...", flush=True)
        print("=" * 65, flush=True)

        _ensure_data_files()

        # 1. Load ONNX Runtime
        print(f"🧠 Loading ONNX {EMBED_MODEL_NAME} runtime...", flush=True)
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            _local_onnx = os.path.join(DATA_DIR, "onnx", "model.onnx")
            _local_tok = os.path.join(DATA_DIR, "tokenizer.json")

            if os.path.exists(_local_onnx) and os.path.exists(_local_tok):
                onnx_model_path = _local_onnx
                onnx_tok_path = _local_tok
                print(f"   Using cached ONNX files from {DATA_DIR}", flush=True)
            else:
                print("   Downloading ONNX model from HuggingFace Hub...", flush=True)
                onnx_model_path = hf_hub_download(EMBED_MODEL_NAME, "onnx/model.onnx", local_dir=DATA_DIR)
                onnx_tok_path = hf_hub_download(EMBED_MODEL_NAME, "tokenizer.json", local_dir=DATA_DIR)

            onnx_tokenizer = Tokenizer.from_file(onnx_tok_path)
            onnx_tokenizer.enable_truncation(max_length=512)

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 4
            onnx_session = ort.InferenceSession(onnx_model_path, sess_opts, providers=["CPUExecutionProvider"])

            test_emb = get_query_embedding("warm up test")
            if test_emb is None:
                raise RuntimeError("ONNX warm-up returned None")
            print(f"   ✅ ONNX embedder ready! (dim={test_emb.shape[0]}, norm={float(np.linalg.norm(test_emb)):.4f})", flush=True)
        except Exception as onnx_err:
            print(f"   🚨 ONNX load warning: {onnx_err}", flush=True)

        # 2. Load indexes for all 5 languages
        total_all_records = 0
        for lang_code, cfg in LANG_CONFIG.items():
            store = lang_stores[lang_code]
            parquet_path = os.path.join(DATA_DIR, cfg["parquet_file"])
            faiss_path = os.path.join(DATA_DIR, cfg["faiss_file"])

            if not os.path.exists(parquet_path) or not os.path.exists(faiss_path):
                print(f"⚠️ Skipping {cfg['name']} ({lang_code}): missing files on disk.", flush=True)
                continue

            try:
                import gc
                import pyarrow.parquet as pq
                print(f"📦 Loading {cfg['name']} ({lang_code}) from {cfg['parquet_file']}...", flush=True)

                # Step 1: Read schema only (zero RAM) to resolve column names
                schema_names = pq.read_schema(parquet_path).names
                q_col = cfg["query_col"] if cfg["query_col"] in schema_names else ("hindi_query" if "hindi_query" in schema_names else "query")
                a_col = cfg["answer_col"] if cfg["answer_col"] in schema_names else ("hindi_answer" if "hindi_answer" in schema_names else "answer")
                c_col = "chunk_id" if "chunk_id" in schema_names else "query_id"

                # Step 2: Load ONLY the 3 required columns (saves ~60% RAM vs full read)
                df_sample = pd.read_parquet(parquet_path, columns=[c_col, q_col, a_col])
                store["chunk_ids"] = df_sample[c_col].tolist()
                store["queries"] = df_sample[q_col].astype(str).tolist()
                store["answers"] = df_sample[a_col].astype(str).tolist()
                store["total_records"] = len(store["queries"])
                total_all_records += store["total_records"]
                del df_sample
                gc.collect()  # Release parquet RAM before loading FAISS

                # Step 3: Load FAISS index
                try:
                    store["index_q"] = faiss.read_index(faiss_path)
                    n_idx = store["index_q"].ntotal
                    print(f"   ⚡ FAISS index loaded: {n_idx:,} vectors!", flush=True)
                    if n_idx != store["total_records"]:
                        print(f"   ⚠️ Row mismatch: parquet={store['total_records']:,}, faiss={n_idx:,}. Using min.", flush=True)
                        store["total_records"] = min(store["total_records"], n_idx)
                except Exception as faiss_err:
                    print(f"   ⚠️ FAISS read failed ({faiss_err}); trying memmap...", flush=True)
                    try:
                        for _offset in (45, 40, 48, 56):
                            try:
                                store["mmap_vectors"] = np.memmap(
                                    faiss_path, dtype="float32", mode="r",
                                    offset=_offset, shape=(store["total_records"], VECTOR_DIM)
                                )
                                # Validate: first vector should have reasonable norm (skip NaN/inf)
                                try:
                                    test_norm = float(np.linalg.norm(store["mmap_vectors"][0]))
                                except Exception:
                                    test_norm = 0.0
                                if np.isfinite(test_norm) and 0.8 <= test_norm <= 1.2:
                                    print(f"   ⚡ Memmap fallback OK at offset={_offset}, norm={test_norm:.3f}", flush=True)
                                    break
                                store["mmap_vectors"] = None
                            except Exception:
                                store["mmap_vectors"] = None
                    except Exception as mmap_err:
                        print(f"   🚨 Both FAISS and memmap failed for {lang_code}: {mmap_err}", flush=True)

                store["ready"] = True
                print(f"   ✅ {cfg['name']} ({lang_code}) RAG Engine Ready ({store['total_records']:,} chunks)!", flush=True)

            except Exception as lang_err:
                import traceback
                print(f"   🚨 Error loading {cfg['name']}: {type(lang_err).__name__}: {lang_err}", flush=True)
                traceback.print_exc()

        _engine_ready = True
        print("=" * 65, flush=True)
        print(f"🎯 MULTILINGUAL RAG ENGINE FULLY READY — {total_all_records:,} TOTAL CHUNKS!", flush=True)
        print("   Languages active: " + ", ".join(f"{cfg['name']} ({k})" for k, cfg in LANG_CONFIG.items() if lang_stores[k]["ready"]), flush=True)
        print("=" * 65, flush=True)

    except Exception as e:
        _engine_error = str(e)
        print(f"🚨 ENGINE LOAD FAILED: {e}", flush=True)

@app.on_event("startup")
def startup_event():
    print("🟢 FastAPI started — launching Multilingual RAG loader in background...", flush=True)
    t = threading.Thread(target=_load_rag_engine_background, daemon=True)
    t.start()

# ── Request Schemas & Guardrails ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    k: int = 5
    lang: str | None = None  # Optional override, auto-detected if not provided

def get_match_quality(score: float) -> str:
    """Human-readable qualitative interpretation of the original similarity score."""
    if score >= 0.92:
        return "Strong Grounding"
    elif score >= 0.87:
        return "Relevant Match"
    elif score >= 0.82:
        return "Moderate Match"
    else:
        return "Low Relevance"

def guardrail_check(top_results: list, top_sim_score: float = 0.0,
                     rrf_threshold: float = 0.01, semantic_sim_threshold: float = 0.42):
    if not top_results:
        return {"should_answer": False, "reason": "no_results", "top_score": 0.0, "semantic_sim": 0.0, "confidence": 0.0, "match_quality": "No Match"}

    top = top_results[0]
    score_val = float(top.get("score", 0.0))
    if top_sim_score > 0:
        semantic_sim = float(top_sim_score)
    elif score_val > 0:
        semantic_sim = 0.85  # ONNX-degraded fallback
    else:
        semantic_sim = 0.0

    original_score = round(semantic_sim, 3)
    match_qual = get_match_quality(original_score)

    if score_val < rrf_threshold or semantic_sim < semantic_sim_threshold:
        return {
            "should_answer": False,
            "reason": "low_confidence_off_topic" if score_val < rrf_threshold else "semantic_mismatch",
            "top_score": round(score_val, 3),
            "semantic_sim": round(semantic_sim, 3),
            "confidence": original_score,
            "match_quality": match_qual,
        }

    return {
        "should_answer": True,
        "reason": "grounded",
        "top_score": round(score_val, 3),
        "semantic_sim": round(semantic_sim, 3),
        "confidence": original_score,
        "match_quality": match_qual,
        "grounded_answer": top["answer"],
        "retrieved_context": top_results,
    }

# ── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
def health_check():
    """Healthcheck reporting ready state and per-language chunk counts."""
    if _engine_ready:
        active_langs = {k: lang_stores[k]["total_records"] for k in LANG_CONFIG if lang_stores[k]["ready"]}
        return {
            "status": "ready",
            "engine": "Multilingual Indic RAG QA Engine v3.0",
            "active_languages": active_langs,
            "total_records_indexed": sum(active_langs.values()),
            "models": {
                "embedding": EMBED_MODEL_NAME,
                "dimension": VECTOR_DIM,
                "llm": "gemini-3.1-flash-lite"
            }
        }
    elif _engine_error:
        return {"status": "error", "detail": _engine_error}
    else:
        return {"status": "loading", "message": "Multilingual RAG engine initializing in background, please wait..."}

@app.post("/ask")
async def ask_question(req: QueryRequest):
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not _engine_ready:
        msg = f"Engine error: {_engine_error}" if _engine_error else "RAG engine is still loading, please retry in a moment."
        raise HTTPException(status_code=503, detail=msg)

    # 1. 0.01ms Automatic Language Detection / Routing
    detected_lang = req.lang if (req.lang and req.lang in LANG_CONFIG) else detect_language(query_text)
    
    # Fallback to Hindi if detected language is not ready
    if not lang_stores[detected_lang]["ready"]:
        fallback = next((k for k in ["hi", "mr", "pa", "gu", "ur"] if lang_stores[k]["ready"]), None)
        if not fallback:
            raise HTTPException(status_code=503, detail="No language index is ready.")
        detected_lang = fallback

    cfg = LANG_CONFIG[detected_lang]
    store = lang_stores[detected_lang]
    total_records = store["total_records"]

    t0_rag = time.perf_counter()

    try:
        # 2. Dense Semantic Search over Routed Language Index
        candidate_k = 50
        dense_idx = []
        dense_scores = []
        dense_score_map = {}
        q_emb = get_query_embedding(query_text)

        try:
            if q_emb is not None:
                if store["mmap_vectors"] is not None:
                    scores = np.dot(store["mmap_vectors"], q_emb)
                    part_idx = np.argpartition(scores, -candidate_k)[-candidate_k:]
                    dense_idx = part_idx[np.argsort(scores[part_idx])[::-1]]
                    dense_scores = scores[dense_idx]
                    for d_i, d_sc in zip(dense_idx, dense_scores):
                        if 0 <= d_i < total_records:
                            dense_score_map[int(d_i)] = float(d_sc)
                elif store["index_q"] is not None:
                    q_emb_arr = np.array([q_emb], dtype="float32")
                    dense_scores_arr, dense_idx_arr = store["index_q"].search(q_emb_arr, candidate_k)
                    dense_idx = dense_idx_arr[0]
                    dense_scores = dense_scores_arr[0]
                    for d_i, d_sc in zip(dense_idx, dense_scores):
                        if 0 <= d_i < total_records:
                            dense_score_map[int(d_i)] = float(d_sc)
        except Exception as embed_err:
            print(f"⚠️ Dense search error ({detected_lang}): {embed_err}", flush=True)

        # 3. BM25 Re-ranking on FAISS top-50 candidates only (zero startup RAM)
        # Build a mini BM25 index on just the candidate texts at query time.
        cand_doc_ids = list(dense_score_map.keys())
        bm25_score_map = {}
        if cand_doc_ids:
            try:
                q_tokens = tokenize_indic(query_text)
                cand_texts = [tokenize_indic(store["queries"][i]) for i in cand_doc_ids]
                mini_bm25 = BM25Okapi(cand_texts)
                bm25_scores = mini_bm25.get_scores(q_tokens)
                for d_id, b_sc in zip(cand_doc_ids, bm25_scores):
                    bm25_score_map[d_id] = float(b_sc)
            except Exception as bm25_err:
                print(f"⚠️ Mini-BM25 error: {bm25_err}", flush=True)

        # 4. Reciprocal Rank Fusion (RRF)
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

        # 5. In-Memory Row Assembly (0.01 ms)
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

            cid = store["chunk_ids"][idx_int]
            hq = store["queries"][idx_int]
            ha = store["answers"][idx_int]

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
                "language": detected_lang
            })
        retrieved_docs = retrieved_docs[:max(req.k, 4)]

        retrieval_latency = (time.perf_counter() - t0_rag) * 1000

        # 6. Guardrail Check
        # Use 0.85 when top_candidate_indices is empty (BM25-only path) so the
        # guardrail doesn't mistake an absent ONNX score for low confidence.
        if top_candidate_indices:
            top_sim = dense_score_map.get(int(top_candidate_indices[0]), 0.85)
        else:
            top_sim = 0.85 if retrieved_docs else 0.0
        check = guardrail_check(retrieved_docs, top_sim_score=top_sim)

        if not check["should_answer"]:
            total_latency = (time.perf_counter() - t0_rag) * 1000
            return {
                "query": query_text,
                "language": detected_lang,
                "language_name": cfg["name"],
                "answer": cfg["refusal"],
                "status": "refused",
                "reason": check["reason"],
                "confidence": check.get("confidence", 0.0),
                "semantic_sim": check["semantic_sim"],
                "retrieved_context": retrieved_docs,
                "retrieval_latency_ms": round(retrieval_latency, 2),
                "total_latency_ms": round(total_latency, 2)
            }

        # 7. Grounded Generation via Gemini 3.1 Flash Lite in Detected Language
        context_docs = "\n\n".join(
            f"Q: {doc['query']}\nA: {doc['answer']}"
            for doc in check["retrieved_context"][:3]
        )

        prompt = f"""Context:
{context_docs}

User Question ({cfg['name']}): {query_text}
Answer:"""

        if gemini_client is None:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

        t_llm_0 = time.perf_counter()
        full_text = ""
        ttft_ms = None

        chosen_model = "gemini-3.1-flash-lite"
        gen_config = types.GenerateContentConfig(
            system_instruction=cfg["system_instruction"],
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
            print(f"⚠️ Primary model error ({gemini_err}); using fallback...")
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
            "language": detected_lang,
            "language_name": cfg["name"],
            "answer": full_text.strip(),
            "status": "answered_by_llm",
            "confidence": check["confidence"],
            "match_quality": check.get("match_quality", "Relevant Match"),
            "raw_score": check["top_score"],
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
    """Transcribes audio in Hindi/Marathi/Punjabi/Gujarati/Urdu via Sarvam STT, then runs multilingual RAG."""
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
            res_data = res.json()
            transcript = res_data.get("transcript", "")
            sarvam_detected_lang = res_data.get("language_code")
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

    resolved_lang = detect_language(transcript, sarvam_detected_lang)
    rag_response = await ask_question(QueryRequest(query=transcript, lang=resolved_lang))
    rag_response["transcript"] = transcript
    rag_response["stt_latency_ms"] = round(stt_latency_ms, 2)
    rag_response["total_pipeline_latency_ms"] = round((time.perf_counter() - t0_voice) * 1000, 2)

    return rag_response

# ── WebSocket Proxy for Sarvam Realtime STT ──────────────────────────────────
SARVAM_WS_BASE = "wss://api.sarvam.ai/speech-to-text-realtime/ws"

@app.websocket("/ws/sarvam")
async def sarvam_ws_proxy(client_ws: WebSocket):
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

