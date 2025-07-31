from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag_system import RAGSystem
import uvicorn
from typing import List, Dict
import os

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio RAG API",
    description="AI-powered assistant for portfolio information",
    version="1.0.0"
)
# Configure CORS
origins = [
    "http://localhost:3000",  # React development
    "http://localhost:3001", 
    "https://shubhamjagtap.com",  # Your production domain
    "https://jshubham17.netlify.app",  # Your Netlify domain
]
# Add CORS middleware to allow requests from your portfolio website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize RAG system (this will load your vector database)
try:
    rag_system = RAGSystem()
    print("✅ RAG system initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize RAG system: {e}")
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
    sources: List[Source]
    status: str
    available_documents: List[str] = []

# API Endpoints
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "Portfolio RAG API is running",
        "rag_system_status": "initialized" if rag_system else "failed",
        "available_documents": rag_system.get_available_documents() if rag_system else []
    }

@app.post("/chat", response_model=QueryResponse)
async def chat_with_portfolio(request: QueryRequest):
    """
    Main chat endpoint for portfolio queries
    
    Example questions:
    - "What programming languages does this person know?"
    - "Tell me about their machine learning projects"
    - "What is their educational background?"
    - "What research have they published?"
    """
    
    if not rag_system:
        raise HTTPException(
            status_code=500, 
            detail="RAG system not initialized properly"
        )
    
    try:
        # Process the question using your RAG system
        result = rag_system.query(request.question)
        
        # Limit sources if requested
        limited_sources = result["sources"][:request.max_sources]
        
        # Convert to response format
        sources = [
            Source(
                content=source["content"],
                source=source["source"],
                page=str(source["page"])
            )
            for source in limited_sources
        ]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            status=result["status"],
            available_documents=rag_system.get_available_documents()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/documents")
async def get_available_documents():
    """Get list of available documents in the knowledge base"""
    
    if not rag_system:
        raise HTTPException(
            status_code=500,
            detail="RAG system not initialized"
        )
    
    return {
        "available_documents": rag_system.get_available_documents(),
        "total_count": len(rag_system.get_available_documents())
    }

@app.get("/sample-questions")
async def get_sample_questions():
    """Get sample questions visitors can ask"""
    
    sample_questions = [
        "What programming languages and technical skills does this person have?",
        "Tell me about their machine learning or AI projects",
        "What is their educational background?",
        "What research has this person conducted?",
        "What certifications do they have?",
        "Tell me about their work experience",
        "What frameworks and tools do they use?",
        "What are their key achievements or accomplishments?"
    ]
    
    return {
        "sample_questions": sample_questions,
        "total_count": len(sample_questions)
    }

# Development server configuration
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Import string instead of app object
        host="0.0.0.0", 
        port=8000,
        reload=True
    )
