import os
import time
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio RAG API",
    description="AI-powered assistant for portfolio information",
    version="1.0.0",
)

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://shubhamjagtap.com",
    "https://jshubham17.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = None
_executor = ThreadPoolExecutor(max_workers=1)

# --- In-memory conversation history ---------------------------------------
# NOTE: This is intentionally simple: a dict in process memory, keyed by
# session_id. It resets on every deploy/restart and does NOT scale across
# multiple server instances. That's fine for this app's current traffic
# (single Railway instance, low volume). If that changes, swap this for a
# real store (Postgres/Redis) without touching the RAGSystem interface.
_MAX_SESSIONS = 500          # simple cap so memory can't grow unbounded
_MAX_TURNS_PER_SESSION = 20  # keep last N messages (user+assistant) per session
_SESSION_TTL_SECONDS = 60 * 60 * 3  # drop sessions untouched for 3 hours

_conversation_store: dict[str, dict] = {}
# shape: { session_id: {"history": [{"role": "user"/"assistant", "content": str}], "last_seen": float} }


def _prune_sessions():
    """Drop stale sessions and enforce a max session count (evict oldest)."""
    now = time.time()
    stale = [sid for sid, s in _conversation_store.items()
             if now - s["last_seen"] > _SESSION_TTL_SECONDS]
    for sid in stale:
        _conversation_store.pop(sid, None)

    if len(_conversation_store) > _MAX_SESSIONS:
        # Evict oldest-touched sessions first
        by_age = sorted(_conversation_store.items(), key=lambda kv: kv[1]["last_seen"])
        for sid, _ in by_age[: len(_conversation_store) - _MAX_SESSIONS]:
            _conversation_store.pop(sid, None)


def _get_history(session_id: str) -> list:
    session = _conversation_store.get(session_id)
    return list(session["history"]) if session else []


def _append_turn(session_id: str, role: str, content: str):
    session = _conversation_store.setdefault(
        session_id, {"history": [], "last_seen": time.time()}
    )
    session["history"].append({"role": role, "content": content})
    session["history"] = session["history"][-_MAX_TURNS_PER_SESSION:]
    session["last_seen"] = time.time()
    _prune_sessions()
# ---------------------------------------------------------------------------


def _init_rag():
    """Run RAG initialization in a thread so it doesn't block startup."""
    global rag_system
    try:
        print("🚀 Starting RAG system initialization...")
        print(f"   GROQ_API_KEY set: {bool(os.getenv('GROQ_API_KEY'))}")
        print(f"   documents/ exists: {os.path.exists('documents')}")
        print(f"   chroma_db/ exists: {os.path.exists('chroma_db')}")
        rag_system = RAGSystem()
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        traceback.print_exc()
        rag_system = None


@app.on_event("startup")
async def startup_event():
    """Start RAG initialization in the background so Railway health checks pass immediately."""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _init_rag)


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    max_sources: int = 3
    session_id: str | None = None  # omit or leave blank for stateless single-turn behavior


class Source(BaseModel):
    content: str
    source: str
    page: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: str
    available_documents: list[str] = []
    session_id: str | None = None


# Endpoints
@app.get("/")
async def health_check():
    return {
        "status": "Portfolio RAG API is running",
        "rag_system_status": "initialized" if rag_system else "failed",
        "available_documents": rag_system.get_available_documents() if rag_system else [],
    }


@app.post("/chat")
async def chat_with_portfolio(request: QueryRequest):
    if not rag_system:
        return JSONResponse(
            status_code=503,
            content={
                "error": "RAG system not initialized. Check server logs for details.",
                "hint": "Ensure GROQ_API_KEY is set and documents/ folder contains PDFs.",
            },
        )
    try:
        if not request.question or not request.question.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "No question provided in the request."},
            )

        print(f"📩 Received question: {request.question} (session_id={request.session_id})")

        history = _get_history(request.session_id) if request.session_id else []
        result = rag_system.query(request.question, history=history)

        # Only persist history for successful answers, and only if a session_id was given
        if request.session_id and result.get("status") == "success":
            _append_turn(request.session_id, "user", request.question)
            _append_turn(request.session_id, "assistant", result["answer"])

        limited_sources = result.get("sources", [])[:request.max_sources]
        print(f"📤 RAG query status: {result.get('status', 'unknown')}")

        return {
            "answer": result["answer"],
            "sources": limited_sources,
            "status": result.get("status", "success"),
            "available_documents": rag_system.get_available_documents(),
            "session_id": request.session_id,
        }
    except Exception as e:
        print(f"❌ Exception in /chat endpoint: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing query: {str(e)}"},
        )


@app.get("/documents")
async def get_available_documents():
    if not rag_system:
        raise HTTPException(500, "RAG system not initialized")
    docs = rag_system.get_available_documents()
    return {"available_documents": docs, "total_count": len(docs)}


@app.get("/sample-questions")
async def get_sample_questions():
    samples = [
        "What programming languages and technical skills does this person have?",
        "Tell me about their machine learning or AI projects",
        "What is their educational background?",
        "What research has this person conducted?",
        "What certifications do they have?",
        "Tell me about their work experience",
        "What frameworks and tools do they use?",
        "What are their key achievements or accomplishments?",
    ]
    return {"sample_questions": samples, "total_count": len(samples)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)