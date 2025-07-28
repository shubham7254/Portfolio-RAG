# Create test_installation.py
import langchain
import chromadb
import groq
from sentence_transformers import SentenceTransformer
import fastapi
import uvicorn

print("✅ All packages imported successfully!")
print(f"LangChain version: {langchain.__version__}")
print(f"ChromaDB available: {chromadb.__version__}")
print(f"Sentence Transformers: Loading test model...")

# Test sentence transformer (optimized for M1)
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Sentence Transformer model loaded successfully!")
