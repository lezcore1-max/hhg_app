import os
import time
import math
import numpy as np
import pandas as pd
import torch
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import faiss

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "lezcore1-max/tilt-rag-data")

def _ensure_data_files():
    """Download FAISS indexes and Parquet from HuggingFace Hub if not present locally."""
    files_needed = ["index_q.faiss", "index_qa.faiss", "qa_pool.parquet"]
    missing = [f for f in files_needed if not os.path.exists(f)]
    if not missing:
        return
    print(f"⬇️  Data files missing: {missing}. Downloading from HuggingFace Hub ({HF_DATASET_REPO})...")
    try:
        from huggingface_hub import hf_hub_download
        for fname in missing:
            print(f"   Downloading {fname}...")
            hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=fname,
                repo_type="dataset",
                local_dir="."
            )
            print(f"   ✅ {fname} ready.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download data files from HuggingFace Hub '{HF_DATASET_REPO}': {e}\n"
            "Place index_q.faiss, index_qa.faiss, qa_pool.parquet in the working directory "
            "or set HF_DATASET_REPO env variable."
        )

# Load environment variables (.env file)
load_dotenv()

# Configure Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables or .env file!")

# Initialize FastAPI Engine
app = FastAPI(
    title="Enterprise Hindi RAG QA Engine",
    description="High-performance hybrid retrieval (FAISS + BM25 + BGE Reranker) powered by Gemini 3.1 Flash Lite",
    version="2.0.0"
)

# Enable CORS for frontend website access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded RAG assets
df = None
index_q = None
index_qa = None
bm25 = None
embed_model = None
reranker_tokenizer = None
reranker_model = None

INDEX_Q_PATH = "index_q.faiss"
INDEX_QA_PATH = "index_qa.faiss"
PARQUET_PATH = "qa_pool.parquet"
EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

@app.on_event("startup")
def startup_event():
    global df, index_q, index_qa, bm25, embed_model, reranker_tokenizer, reranker_model
    print("=" * 60)
    print("🚀 INITIALIZING HINDI RAG ENGINE (NATIVE FAISS + PARQUET)...")
    print("=" * 60)

    # 0. Auto-download data files from HuggingFace Hub if needed
    _ensure_data_files()

    # 1. Load Parquet Data
    print(f"📦 Loading dataset from {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"   Loaded {len(df):,} QA records.")

    # 2. Load Native FAISS Indexes
    print(f"⚡ Loading native FAISS indexes...")
    print(f"⚡ Loading native FAISS indexes ({INDEX_Q_PATH}, {INDEX_QA_PATH})...")
    index_q = faiss.read_index(INDEX_Q_PATH)
    index_qa = faiss.read_index(INDEX_QA_PATH)

    # 3. Build BM25 Index
    print("🔍 Building BM25 keyword index...")
    tokenized_queries = [q.split() for q in df["query"].tolist()]
    bm25 = BM25Okapi(tokenized_queries)

    # 4. Load SentenceTransformer Model
    print(f"🧠 Loading Embedding Model ({EMBED_MODEL_NAME})...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 5. Load Cross-Encoder Re-ranker
    print(f"🎯 Loading Cross-Encoder Re-ranker ({RERANKER_MODEL_NAME})...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME).to("cpu")
    reranker_model.eval()

    print("=" * 60)
    print("✅ HINDI RAG ENGINE READY ON http://localhost:8000")
    print("=" * 60)

class QueryRequest(BaseModel):
    query: str
    k: int = 5

def guardrail_check(query_text: str, top_results: list, score_threshold: float = 0.5, semantic_sim_threshold: float = 0.75):
    if not top_results:
        return {"should_answer": False, "reason": "no_results"}

    top = top_results[0]
    if top["score"] < score_threshold:
        return {"should_answer": False, "reason": "low_confidence_off_topic", "top_score": top["score"]}

    q_emb = embed_model.encode([f"query: {query_text}"], normalize_embeddings=True)
    mq_emb = embed_model.encode([f"passage: {top['query']}"], normalize_embeddings=True)
    semantic_sim = float(np.dot(q_emb[0], mq_emb[0]))

    if semantic_sim < semantic_sim_threshold:
        return {
            "should_answer": False,
            "reason": "semantic_mismatch",
            "top_score": top["score"],
            "semantic_sim": semantic_sim,
        }

    return {
        "should_answer": True,
        "reason": "grounded",
        "top_score": top["score"],
        "semantic_sim": semantic_sim,
        "grounded_answer": top["answer"],
        "retrieved_context": top_results,
    }

@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine": "Hindi RAG QA Engine v2.0",
        "records_indexed": len(df) if df is not None else 0,
        "models": {
            "embedding": EMBED_MODEL_NAME,
            "reranker": RERANKER_MODEL_NAME,
            "llm": "gemini-3.1-flash-lite"
        }
    }

@app.post("/ask")
def ask_question(req: QueryRequest):
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if index_q is None or embed_model is None:
        raise HTTPException(status_code=500, detail="RAG Engine components are not initialized.")

    t0_rag = time.perf_counter()

    try:
        # 1. Embed query
        q_emb = embed_model.encode([f"query: {query_text}"], normalize_embeddings=True)

        # 2. Hybrid Search (Dense FAISS + BM25)
        rerank_k = max(req.k, 5)
        dense_scores, dense_idx = index_q.search(np.array(q_emb, dtype="float32"), rerank_k)
        dense_idx = dense_idx[0]

        bm25_scores = np.array(bm25.get_scores(query_text.split()))

        # Reciprocal Rank Fusion (RRF)
        k_rrf = 60
        rrf_scores = np.zeros(len(df))
        for rank, original_idx in enumerate(dense_idx):
            rrf_scores[original_idx] += 1 / (k_rrf + rank + 1)

        bm25_top_idx = np.argsort(bm25_scores)[::-1][:rerank_k]
        for rank, original_idx in enumerate(bm25_top_idx):
            rrf_scores[original_idx] += 1 / (k_rrf + rank + 1)

        top_candidate_indices = np.argsort(rrf_scores)[::-1][:rerank_k]
        candidate_results = [df.iloc[int(i)] for i in top_candidate_indices]

        # 3. Cross-Encoder Re-ranking
        passages = [f"{doc['query']} {doc['answer']}" for doc in candidate_results]
        features = reranker_tokenizer(
            [query_text] * len(passages), passages,
            padding=True, truncation=True, return_tensors='pt'
        ).to('cpu')

        with torch.no_grad():
            logits = reranker_model(**features).logits.view(-1)
        raw_scores = logits.tolist()
        probs = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]

        retrieved_docs = []
        for i, (raw, prob) in enumerate(zip(raw_scores, probs)):
            cand = candidate_results[i].to_dict()
            retrieved_docs.append({
                "query": cand["query"],
                "matched_question": cand["query"],  # Map query to matched_question for compatibility
                "answer": cand["answer"],
                "score": float(prob),
                "raw_score": float(raw)
            })

        retrieved_docs.sort(key=lambda x: x["score"], reverse=True)
        retrieved_docs = retrieved_docs[:req.k]

        retrieval_latency = (time.perf_counter() - t0_rag) * 1000

        # 4. Guardrail Check
        check = guardrail_check(query_text, retrieved_docs)

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

        # 5. Build Grounded Prompt for Gemini LLM
        context_docs = ""
        for i, doc in enumerate(check["retrieved_context"]):
            context_docs += f"Document {i+1}:\nQuestion: {doc['query']}\nAnswer: {doc['answer']}\n\n"

        prompt = f"""You are an expert Hindi AI Assistant. Based solely on the provided documents, answer the user's question accurately in natural Hindi.
If the answer is not present in the documents, state clearly that you cannot find the answer based on the provided information.
Do not use any external knowledge outside of the provided documents.

User Question: {query_text}

Provided Documents:
{context_docs}

Your Answer:
"""

        # 6. Call Gemini 3.1 Flash Lite LLM
        model_gemini = genai.GenerativeModel('gemini-3.1-flash-lite')
        response = model_gemini.generate_content(prompt)

        total_latency = (time.perf_counter() - t0_rag) * 1000

        return {
            "query": query_text,
            "answer": response.text.strip(),
            "status": "answered_by_llm",
            "confidence": check["top_score"],
            "semantic_sim": check["semantic_sim"],
            "retrieved_context": check["retrieved_context"],
            "retrieval_latency_ms": round(retrieval_latency, 2),
            "total_latency_ms": round(total_latency, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import UploadFile, File

@app.post("/voice-ask")
async def voice_ask_question(file: UploadFile = File(...)):
    """Accepts an audio file upload (WAV/MP3), transcribes via SarvamAI SDK (saaras:v3), and runs RAG + Gemini."""
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=500, detail="SARVAM_API_KEY is not set in environment or .env file.")

    t0_voice = time.perf_counter()
    stt_t0 = time.perf_counter()
    transcript = ""

    # Save uploaded bytes to a temporary audio file on disk
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Attempt transcription via SarvamAI SDK
        try:
            from sarvamai import SarvamAI
            client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
            with open(temp_path, "rb") as audio_file:
                stt_res = client.speech_to_text.transcribe(
                    file=audio_file,
                    model="saaras:v3",
                    mode="transcribe"
                )
            # Retrieve transcript property or dict key
            transcript = getattr(stt_res, "transcript", "") or (stt_res.get("transcript", "") if isinstance(stt_res, dict) else "")

        except ImportError:
            # Fallback to direct HTTP request if sarvamai SDK is not installed
            import requests
            url = "https://api.sarvam.ai/speech-to-text"
            headers = {"api-subscription-key": SARVAM_API_KEY}
            with open(temp_path, "rb") as audio_file:
                files = {"file": (file.filename, audio_file, file.content_type or "audio/wav")}
                data = {"model": "saaras:v3", "mode": "transcribe"}
                res = requests.post(url, headers=headers, files=files, data=data)
            if res.status_code == 200:
                transcript = res.json().get("transcript", "")
            else:
                raise HTTPException(status_code=500, detail=f"Sarvam STT failed [{res.status_code}]: {res.text}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice processing error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    stt_latency_ms = (time.perf_counter() - stt_t0) * 1000
    transcript = transcript.strip()

    if not transcript:
        return {"status": "error", "stage": "stt", "reason": "empty transcript"}

    # Process transcript with RAG + Gemini pipeline
    rag_response = ask_question(QueryRequest(query=transcript))
    rag_response["transcript"] = transcript
    rag_response["stt_latency_ms"] = round(stt_latency_ms, 2)
    rag_response["total_pipeline_latency_ms"] = round((time.perf_counter() - t0_voice) * 1000, 2)

    return rag_response

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
