import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load API keys
load_dotenv('secrets.txt')
load_dotenv()  # fallback to .env

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
# Must match the embedding model used in document_processor.py when the
# chroma_db/ folder was built — otherwise similarity search will be nonsense.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class RAGSystem:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please add it to your environment or secrets.txt file."
            )
        # Use the fast 8b model — higher TPM limit, lower latency
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.llm = ChatGroq(api_key=groq_api_key, model=model, temperature=0.1)

        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError(
                f"Chroma vector store not found at '{CHROMA_DIR}'. "
                "Run document_processor.py first to build it."
            )

        # Same embedding model used to build the store — required for
        # similarity search to produce meaningful results.
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
        )

        collection_count = self.vectorstore._collection.count()
        print(f"✅ Loaded Chroma vector store with {collection_count} chunks")

        # Prompt 0: rewrite a follow-up question into a self-contained one using history
        self.rewrite_prompt = PromptTemplate(
            template="""Given the recent conversation and a new question, rewrite the new \
question so it is fully self-contained and makes sense without needing the conversation \
history. If the question is already self-contained, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation history:
{history}

New question: {question}

Rewritten question:""",
            input_variables=["history", "question"],
        )

        # Prompt: answer using retrieved context + conversation history
        self.answer_prompt = PromptTemplate(
            template="""You are a smart AI assistant on a personal portfolio website. \
A visitor is asking about the person who owns this portfolio.
Use the context below — drawn from their resume, project reports, and research — \
to give a helpful, intelligent answer.

Recent conversation (for context only, may be empty):
{history}

Context:
{context}

Visitor's question: {question}

Guidelines:
- Answer ONLY what was specifically asked — do not summarize unrelated projects, \
skills, or background that the question didn't ask about
- If the question is a follow-up (e.g. "tell me more about that", "the first one"), \
give a focused deep-dive on that ONE specific thing only — do not re-list everything \
mentioned earlier in the conversation
- Answer in a natural, confident tone (e.g. "He has..." or "They have...")
- Synthesize information into a coherent answer — don't just bullet-point raw facts
- Use the recent conversation only to understand what the visitor means; don't repeat it back
- Do NOT mention document names, file names, or where the info came from
- Do NOT reproduce code snippets unless directly asked about code
- If the context is insufficient, say so briefly and helpfully
- Keep the answer concise, focused, and conversational — suitable for a chat widget \
(a few sentences to one short paragraph, not an exhaustive report)

Answer:""",
            input_variables=["history", "context", "question"],
        )

    @staticmethod
    def _format_history(history: list, max_turns: int = 6) -> str:
        """Turn a list of {'role': ..., 'content': ...} dicts into a readable transcript.
        Keeps only the most recent `max_turns` messages to control prompt size."""
        if not history:
            return "(no previous conversation)"
        recent = history[-max_turns:]
        lines = []
        for turn in recent:
            role = "Visitor" if turn.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {turn.get('content', '')}")
        return "\n".join(lines)

    def _rewrite_question(self, question: str, history_text: str) -> str:
        """Use the LLM to make a follow-up question self-contained, given history.
        Skips the call entirely if there's no history yet (saves a request)."""
        if history_text == "(no previous conversation)":
            return question
        try:
            prompt = self.rewrite_prompt.format(history=history_text, question=question)
            response = self.llm.invoke(prompt)
            rewritten = response.content.strip()
            return rewritten if rewritten else question
        except Exception:
            # If rewriting fails for any reason, fall back to the original question
            return question

    def _retrieve(self, question: str, k: int = 5):
        """
        Semantic retrieval via Chroma: embeds the question and finds the
        k chunks whose meaning is closest, regardless of exact word overlap.
        """
        results = self.vectorstore.similarity_search(question, k=k)
        return results  # list of LangChain Document objects (.page_content, .metadata)

    def query(self, question: str, history: list = None) -> dict:
        """
        Full RAG pipeline with optional multi-turn memory:
          question (+history) → rewrite → Chroma semantic search → LLM answer

        `history` is a list of {"role": "user"|"assistant", "content": str}, oldest first.
        Pass None or [] for a stateless, single-turn call (backwards compatible).
        """
        try:
            history_text = self._format_history(history or [])

            # Make follow-up questions self-contained — use this same rewritten
            # version for BOTH retrieval and the final answer, so the model
            # answers the specific, resolved question rather than the raw
            # (possibly vague) follow-up phrasing.
            search_question = self._rewrite_question(question, history_text)

            relevant_docs = self._retrieve(search_question)
            context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

            answer_prompt = self.answer_prompt.format(
                history=history_text, context=context, question=search_question
            )
            response = self.llm.invoke(answer_prompt)

            sources = []
            seen = set()
            for doc in relevant_docs:
                source = doc.metadata.get("source", "unknown")
                key = os.path.basename(source)
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "source": source,
                        "page": str(doc.metadata.get("page", "")),
                    })
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
        """Return unique document names from the vector store."""
        all_docs = self.vectorstore.get()
        sources = {
            os.path.basename(meta.get("source", ""))
            for meta in all_docs.get("metadatas", [])
            if meta.get("source")
        }
        return list(sources)