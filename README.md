# Tilt RAG — Hindi Voice & Text QA Engine 🎙️⚡

> Voice-native Hindi RAG system: speak or type a question in Hindi, get a grounded answer in real-time with measured latency across Speech-to-Text (Sarvam AI), Hybrid Retrieval (FAISS + BM25), and LLM Generation (Gemini 3.1 Flash Lite).

---

## 🏗️ System Architecture

```
[ User (Voice/Text) ] 
        │
        ├── Audio File (.webm/.wav) ──► Sarvam AI (saaras:v3) ──► Transcript
        │
        └── Query Text ──► HuggingFace Inference API (BAAI/bge-m3) ──► 1024D Query Vector
                                    │
                                    ├── FAISS Index (Dense Vector Search) ──┐
                                    │                                       ├──► RRF Fusion ──► Top Matches
                                    └── BM25 Index (Sparse Keyword Search) ─┘
                                                                │
                                                                ▼
                                                Gemini 3.1 Flash Lite LLM
                                                                │
                                                                ▼
                                            Grounded Hindi Answer + Latencies
```

---

## 📁 Repository Structure

```
.
├── app.py                   # FastAPI backend server & RAG engine
├── requirements.txt         # Python backend dependencies
├── Dockerfile               # Production Docker container setup
├── railway.toml             # Railway deployment config
├── .env.example             # Template for API keys
└── hhg-main/hhg-main/       # Vite + React + TanStack Router frontend SPA
    ├── src/
    │   ├── components/      # UI components (MicOrb, PageShell, etc.)
    │   ├── lib/api.ts       # Backend API integration client
    │   └── routes/          # Application routes (/demo, /benchmarks, etc.)
    └── package.json
```

---

## ⚡ Quick Start: Running End-to-End Locally

Follow these step-by-step instructions to run both the **Backend** and **Frontend** on your local machine.

### Prerequisites
- **Python 3.10+** installed
- **Node.js 18+** and `npm` installed
- API Keys:
  - **Sarvam AI API Key** (for Hindi Speech-to-Text) — Get one at [Sarvam.ai](https://www.sarvam.ai)
  - **Gemini API Key** (for Grounded Answer Generation) — Get one at [Google AI Studio](https://aistudio.google.com)

---

### Step 1: Backend Setup (FastAPI RAG Engine)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lezcore1-max/hhg_app.git
   cd hhg_app
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file for API keys**:
   ```bash
   cp .env.example .env
   ```
   Open `.env` in a text editor and add your keys:
   ```env
   SARVAM_API_KEY=sk_your_sarvam_key_here
   GEMINI_API_KEY=your_gemini_key_here
   HF_TOKEN=hf_your_huggingface_token_here
   ```

5. **Index Data Files**:
   The RAG engine requires 3 index files to operate:
   - `index_q.faiss` (FAISS vector index for query matching)
   - `index_qa.faiss` (FAISS vector index for QA matching)
   - `qa_pool.parquet` (13,312 record dataset pool)

   > 💡 **Automatic Download**: When you start `app.py`, it automatically downloads missing data files from Hugging Face Hub dataset [`lezcore1-max/tilt-rag-data`](https://huggingface.co/datasets/lezcore1-max/tilt-rag-data) directly into the directory!

6. **Start the Backend Server**:
   ```bash
   python app.py
   ```
   - Server running at: **`http://localhost:8000`**
   - Interactive Swagger API Docs at: **`http://localhost:8000/docs`**

---

### Step 2: Frontend Setup (Vite React SPA)

1. **Navigate to the frontend directory**:
   ```bash
   cd hhg-main/hhg-main
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Configure Backend URL**:
   Create a `.env.local` file inside `hhg-main/hhg-main`:
   ```env
   VITE_BACKEND_URL=http://localhost:8000
   ```

4. **Start the Frontend Development Server**:
   ```bash
   npm run dev
   ```
   - Application running at: **`http://localhost:5173`**

---

## 🧪 Testing the Live Voice & Text QA Demo

1. Open your browser to **`http://localhost:5173/demo`**.
2. **Voice QA**:
   - Click the central **Yellow Mic Orb**.
   - Speak your question in Hindi (e.g. *"मगरमच्छ का लिंग कैसे निर्धारित होता है?"*).
   - Click again to stop speaking (or wait 12s for auto-stop).
   - Watch the pipeline execute stage-by-stage:
     1. **Speech-to-Text**: Sarvam AI transcribes audio to text.
     2. **Hybrid Retrieval**: BGE-M3 query vector + BM25 keyword search retrieve top matching context.
     3. **Grounded Generation**: Gemini 3.1 Flash Lite returns a grounded answer.
     4. **Latencies**: Measured latency displayed for STT, Retrieval, and Time-To-First-Token (TTFT).

3. **Text QA**:
   - Type any Hindi query into the input box and click **"पूछें"**.

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check & loaded model metadata |
| `POST` | `/ask` | Text Query → FAISS + BM25 Retrieval → Gemini Answer |
| `POST` | `/voice-ask` | Audio File Upload → Sarvam STT → Hybrid RAG → Gemini Answer |

---

## 🚀 Deployment Guide

### Deploy Backend (Railway / Hugging Face Spaces)
1. Push `app.py`, `requirements.txt`, and `Dockerfile` to your GitHub repo.
2. In Railway or HF Spaces, add environment variables:
   - `SARVAM_API_KEY`
   - `GEMINI_API_KEY`
   - `HF_TOKEN`
3. Railway automatically detects Dockerfile and deploys the port.

### Deploy Frontend (Vercel)
1. Import `hhg_appi` repository into Vercel.
2. Set Environment Variable:
   - `VITE_BACKEND_URL = https://your-backend-railway-url.up.railway.app`
3. Framework Preset: **Vite**
4. Output Directory: **`dist`**

---

## 📄 License

MIT License. Developed for high-performance Hindi Voice RAG benchmarks.
