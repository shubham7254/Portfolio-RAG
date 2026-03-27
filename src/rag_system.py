import os
import json
import requests
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load API keys
load_dotenv('secrets.txt')
load_dotenv()  # fallback to .env

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarity between query_vec and each row of matrix."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return (matrix / norms) @ q


class RAGSystem:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please add it to your environment or secrets.txt file."
            )

        self.hf_token = os.getenv("HF_TOKEN", "")
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1,
        )

        emb_path = os.path.join(VECTORSTORE_DIR, "embeddings.npy")
        doc_path = os.path.join(VECTORSTORE_DIR, "documents.json")
        if not os.path.exists(emb_path) or not os.path.exists(doc_path):
            raise FileNotFoundError(
                f"Vectorstore files not found in '{VECTORSTORE_DIR}/'. "
                "Expected embeddings.npy and documents.json."
            )

        self.embeddings_matrix = np.load(emb_path)  # (N, 384)
        with open(doc_path, "r") as f:
            self.documents = json.load(f)

        print(f"✅ Loaded {len(self.documents)} chunks from vectorstore")

        self.prompt = PromptTemplate(
            template="""You are a smart AI assistant on a personal portfolio website. \
A visitor is asking about the person who owns this portfolio.

Use the context below — drawn from their resume, project reports, and research — \
to give a helpful, intelligent answer. Synthesize the information rather than just listing it.

Context:
{context}

Visitor's question: {question}

Guidelines:
- Answer in a natural, confident tone (e.g. "He has..." or "They have...")
- Synthesize information into a coherent answer — don't just bullet-point raw facts
- Do NOT mention document names, file names, or where the info came from
- Do NOT reproduce code snippets unless directly asked about code
- If the context is insufficient, say so briefly and helpfully
- Keep the answer focused and conversational — suitable for a chat widget on a portfolio site

Answer:""",
            input_variables=["context", "question"],
        )

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a query string using the HuggingFace Inference API."""
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        response = requests.post(
            HF_EMBED_URL,
            headers=headers,
            json={"inputs": text},
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()
        # HF returns (seq_len, dim) for feature-extraction — mean pool to (dim,)
        arr = np.array(result, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        return arr

    def _retrieve(self, query: str, k: int = 6):
        """Return top-k most similar documents via cosine similarity."""
        query_vec = self._embed_query(query)
        scores = _cosine_similarity(query_vec, self.embeddings_matrix)
        top_k = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_k]

    def query(self, question: str) -> dict:
        """Query the system with a user question."""
        try:
            relevant_docs = self._retrieve(question)
            context = "\n\n---\n\n".join(doc["text"] for doc in relevant_docs)
            prompt_text = self.prompt.format(context=context, question=question)
            response = self.llm.invoke(prompt_text)
            sources = [
                {
                    "content": doc["text"][:200] + "...",
                    "source": doc["source"],
                    "page": doc["page"],
                }
                for doc in relevant_docs[:3]
            ]
            return {"answer": response.content, "sources": sources, "status": "success"}
        except Exception as e:
            return {"answer": f"I apologize, but I encountered an error: {str(e)}", "sources": [], "status": "error"}

    def get_available_documents(self) -> list:
        """Return unique document names from the vectorstore."""
        return list({os.path.basename(doc["source"]) for doc in self.documents})
