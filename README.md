# Portfolio RAG Chatbot

A production-deployed AI chatbot that answers questions about my portfolio — projects, skills, experience, and background — using a **Retrieval-Augmented Generation (RAG)** pipeline with hybrid search.

🚀 **Live Demo:** [YOUR-RAILWAY-URL-HERE]

---

## What It Does

Instead of a static portfolio page, this chatbot lets anyone ask natural language questions like:

- *"What AI projects has Shubham built?"*
- *"Does he have experience with LLMs and RAG?"*
- *"What cloud platforms has he worked with?"*
- *"Tell me about his research experience."*

The system retrieves the most relevant context from my portfolio documents and generates grounded, accurate answers using an LLM — no hallucinations about things not in the knowledge base.

---

## Architecture

```
User Query
    │
    ▼
Hybrid Search (BM25 + ChromaDB Vector Search)
    │
    ▼
Context Retrieval & Reranking
    │
    ▼
LangChain Orchestration → Groq LLM (llama3 / mixtral)
    │
    ▼
Grounded Response via FastAPI
```

**Hybrid Search** combines:
- **BM25** (keyword-based, `rank-bm25`) — catches exact term matches
- **ChromaDB** (semantic vector search) — catches meaning-based matches

This outperforms pure vector search on queries with specific technical terms (e.g., exact project names, tools, frameworks).

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Orchestration | LangChain Core, LangChain-Groq |
| LLM Provider | Groq API (llama3 / mixtral), OpenAI API |
| Vector Database | ChromaDB (persisted local store) |
| Hybrid Search | rank-bm25 + ChromaDB semantic search |
| API Server | FastAPI + Uvicorn |
| Data Validation | Pydantic |
| Containerization | Docker (Python 3.12-slim) |
| Deployment | Railway |
| Config Management | python-dotenv (.env) |

---

## Project Structure

```
Portfolio-RAG/
├── src/                    # Application source code
│   └── main.py             # FastAPI app entry point
├── documents/              # Portfolio knowledge base (source documents)
├── chroma_db/              # Persisted ChromaDB vector store
├── vectorstore/            # Vectorstore artifacts
├── Dockerfile              # Container definition
├── Procfile                # Railway process config
├── railway.json            # Railway deployment config
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── pyproject.toml          # Project metadata
```

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/shubham7254/Portfolio-RAG.git
cd Portfolio-RAG
```

**2. Set up environment variables**
```bash
cp .env.example .env
# Add your API keys to .env
```

```env
GROQ_API_KEY=your-groq-api-key
OPENAI_API_KEY=your-openai-api-key
```

**3. Run with Docker**
```bash
docker build -t portfolio-rag .
docker run -p 8000:8000 --env-file .env portfolio-rag
```

**4. Or run locally with Python**
```bash
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

**5. Test the API**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What projects has Shubham built with LLMs?"}'
```

API docs available at: `http://localhost:8000/docs`

---

## Key Design Decisions

**Why hybrid search over pure vector search?**
Pure vector search can miss exact term matches (e.g., "YOLOv8", "DeBERTa", specific project names). BM25 catches these precisely. Combining both with a reranking step gives better retrieval across all query types.

**Why Groq?**
Groq's inference speed (500+ tokens/sec on llama3) enables sub-second response times critical for a live chatbot experience.

**Why ChromaDB with persistence?**
The vectorstore is built once from portfolio documents and persisted to disk — no re-embedding on each startup, keeping cold start times low on Railway's free tier.

**Why FastAPI?**
Async-native, Pydantic validation out of the box, and automatic OpenAPI docs — production-grade API serving with minimal boilerplate.

---

## Deployment

Deployed on **Railway** via Docker. The `Dockerfile` uses a `python:3.12-slim` base image and respects Railway's dynamic `$PORT` environment variable:

```dockerfile
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

CI/CD is handled by Railway's GitHub integration — every push to `main` triggers an automatic redeploy.

---

## Skills Demonstrated

- End-to-end RAG pipeline design and production deployment
- Hybrid search (BM25 + semantic vector retrieval)
- LLM orchestration with LangChain
- Vector database management (ChromaDB)
- FastAPI microservice architecture
- Docker containerization and cloud deployment (Railway)
- Secure credential management with environment variables

---

## Author

**Shubham Jagtap**
M.S. Artificial Intelligence — University of Michigan

[LinkedIn](https://linkedin.com/in/jshubham17) · [GitHub](https://github.com/shubham7254) · [Email](mailto:jshubham@umich.edu)
