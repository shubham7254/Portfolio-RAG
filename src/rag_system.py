import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

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
                "Expected documents.json in vectorstore directory."
            )

        with open(doc_path, "r") as f:
            documents = json.load(f)

        # Pre-build full context string once at startup — no retrieval step needed.
        # Groq's llama-3.3-70b-versatile supports 128K tokens; all 70 chunks
        # total ~24K tokens so the entire knowledge base fits in one LLM call.
        self.full_context = "\n\n---\n\n".join(doc["text"] for doc in documents)
        self.document_sources = [
            {"source": doc["source"], "page": doc["page"]}
            for doc in documents
        ]

        print(
            f"✅ Loaded {len(documents)} chunks "
            f"({len(self.full_context):,} chars) into full-context RAG"
        )

        self.prompt = PromptTemplate(
            template="""You are a smart AI assistant on a personal portfolio website. \
A visitor is asking about the person who owns this portfolio.

Below is the complete knowledge base — resume, project reports, and research — \
about this person. Read it carefully and answer the visitor's question with \
full understanding of the context.

Knowledge base:
{context}

Visitor's question: {question}

Guidelines:
- Answer in a natural, confident tone (e.g. "He has..." or "They have...")
- Synthesize information into a coherent, focused answer
- Do NOT mention document names, file names, or where the info came from
- Do NOT reproduce code snippets unless directly asked about code
- If the knowledge base doesn't contain enough info, say so briefly and helpfully
- Keep the answer conversational — suitable for a chat widget on a portfolio site

Answer:""",
            input_variables=["context", "question"],
        )

    def query(self, question: str) -> dict:
        """Query the RAG system. The LLM receives the full knowledge base and
        semantically understands the question to produce an accurate answer."""
        try:
            prompt_text = self.prompt.format(
                context=self.full_context,
                question=question,
            )
            response = self.llm.invoke(prompt_text)
            # Return the top 3 unique source documents as references
            seen = set()
            sources = []
            for s in self.document_sources:
                key = os.path.basename(s["source"])
                if key not in seen:
                    seen.add(key)
                    sources.append(s)
                if len(sources) == 3:
                    break

            return {
                "answer": response.content,
                "sources": sources,
                "status": "success",
            }
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "sources": [],
                "status": "error",
            }

    def get_available_documents(self) -> list:
        """Return unique document names from the vectorstore."""
        return list({os.path.basename(s["source"]) for s in self.document_sources})
