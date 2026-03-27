import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from rank_bm25 import BM25Okapi

# Load API keys
load_dotenv('secrets.txt')
load_dotenv()  # fallback to .env

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")


class RAGSystem:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please add it to your environment or secrets.txt file."
            )

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1,
        )

        doc_path = os.path.join(VECTORSTORE_DIR, "documents.json")
        if not os.path.exists(doc_path):
            raise FileNotFoundError(
                f"Vectorstore file not found: '{doc_path}'. "
                "Expected documents.json."
            )

        with open(doc_path, "r") as f:
            self.documents = json.load(f)

        # Build BM25 index from document text — no model loading, no external API
        tokenized_corpus = [doc["text"].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"✅ Loaded {len(self.documents)} chunks into BM25 index")

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

    def _retrieve(self, query: str, k: int = 6):
        """Return top-k most relevant documents via BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
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
