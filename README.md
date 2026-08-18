# Tilt RAG — Hindi Voice QA Backend

> Voice-native RAG engine: speak a Hindi question, get a grounded answer in ~173ms P50.

## Stack
- **FastAPI** — REST API server
- **FAISS** — native vector index (`index_q.faiss`, `index_qa.faiss`)
- **BAAI/bge-m3** — multilingual dense embeddings
- **BAAI/bge-reranker-v2-m3** — cross-encoder reranker
- **BM25** — sparse keyword retrieval (hybrid RRF fusion)
- **Sarvam AI `saaras:v3`** — Hindi speech-to-text
- **Gemini** — grounded answer generation

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ask` | Text query → RAG answer |
| POST | `/voice-ask` | Audio file → Sarvam STT → RAG answer |
| GET | `/docs` | Auto-generated Swagger UI |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add API keys
```bash
cp .env.example .env
# Fill in your SARVAM_API_KEY and GEMINI_API_KEY in .env
```

### 3. Data Indexes (`index_q.faiss`, `index_qa.faiss`, `qa_pool.parquet`)

The RAG engine requires 3 index files to run:
- `index_q.faiss` — FAISS vector index of 13,312 query embeddings
- `index_qa.faiss` — FAISS vector index of question+answer embeddings
- `qa_pool.parquet` — Full dataset pool containing queries, contexts, and answers

**⚡ Automatic Download (Default):**
When `app.py` runs, it automatically downloads all 3 files from HuggingFace Hub repository [`lezcore1-max/tilt-rag-data`](https://huggingface.co/datasets/lezcore1-max/tilt-rag-data) on startup if they are not already present locally.

**📦 Manual Placement (Optional):**
If you prefer offline setup, download the 3 files from [`lezcore1-max/tilt-rag-data`](https://huggingface.co/datasets/lezcore1-max/tilt-rag-data) and place them directly in the root project directory alongside `app.py`.

### 4. Run
```bash
python app.py
# Server starts at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SARVAM_API_KEY` | Sarvam AI API key for STT |
| `GEMINI_API_KEY` | Google Gemini API key |
| `PORT` | Server port (default: 8000, HF Spaces needs 7860) |
