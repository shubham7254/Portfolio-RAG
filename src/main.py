import os
import traceback
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
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = None

@app.on_event("startup")
async def startup_event():
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

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    max_sources: int = 3

class Source(BaseModel):
    content: str
    source: str
    page: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: str
    available_documents: list[str] = []

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

        print(f"📩 Received question: {request.question}")

        result = rag_system.query(request.question)
        limited_sources = result.get("sources", [])[:request.max_sources]

        print(f"📤 RAG query status: {result.get('status', 'unknown')}")

        return {
            "answer": result["answer"],
            "sources": limited_sources,
            "status": result.get("status", "success"),
            "available_documents": rag_system.get_available_documents(),
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
