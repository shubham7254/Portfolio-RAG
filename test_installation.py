import langchain
import chromadb
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import fastapi
import uvicorn

print("✅ All packages imported successfully!")
print(f"LangChain version: {langchain.__version__}")
print(f"ChromaDB available: {chromadb.__version__}")
print("✅ Groq LLM imported successfully")
print("Sentence Transformers: Loading test model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Sentence Transformer model loaded successfully!")
