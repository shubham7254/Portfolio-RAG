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
        # Use the fast 8b model — higher TPM limit, lower latency
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.llm = ChatGroq(api_key=groq_api_key, model=model, temperature=0.1)

        doc_path = os.path.join(VECTORSTORE_DIR, "documents.json")
        if not os.path.exists(doc_path):
            raise FileNotFoundError(
                f"Vectorstore file not found: '{doc_path}'. "
                "Expected documents.json in vectorstore directory."
            )
        with open(doc_path, "r") as f:
            self.documents = json.load(f)

        # Build BM25 index over document text
        tokenized_corpus = [doc["text"].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ Loaded {len(self.documents)} chunks into BM25 index")

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

        # Prompt 1: extract search keywords from the user question (tiny call)
        self.keyword_prompt = PromptTemplate(
            template="""Extract 6-10 specific search keywords from this question that would \
help find relevant information in a resume or portfolio.
Return ONLY the keywords separated by spaces, nothing else.
Question: {question}
Keywords:""",
            input_variables=["question"],
        )

        # Prompt 2: answer using retrieved context + conversation history
        self.answer_prompt = PromptTemplate(
            template="""You are a smart AI assistant on a personal portfolio website. \
A visitor is asking about the person who owns this portfolio.
Use the context below — drawn from their resume, project reports, and research — \
to give a helpful, intelligent answer.

Recent conversation (for context, may be empty):
{history}

Context:
{context}

Visitor's question: {question}

Guidelines:
- Answer in a natural, confident tone (e.g. "He has..." or "They have...")
- Synthesize information into a coherent answer — don't just bullet-point raw facts
- Use the recent conversation only to understand what the visitor means; don't repeat it back
- Do NOT mention document names, file names, or where the info came from
- Do NOT reproduce code snippets unless directly asked about code
- If the context is insufficient, say so briefly and helpfully
- Keep the answer focused and conversational — suitable for a chat widget

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

    def _expand_and_retrieve(self, question: str, k: int = 8):
        """
        Two-step semantic retrieval:
        1. LLM expands the question into domain-specific keywords (~100 tokens)
        2. BM25 retrieves top-k chunks using those keywords
        """
        # Step 1: semantic query expansion via LLM
        keyword_prompt = self.keyword_prompt.format(question=question)
        keywords_response = self.llm.invoke(keyword_prompt)
        expanded_query = keywords_response.content.strip()

        # Step 2: BM25 retrieval with expanded keywords
        tokenized_query = expanded_query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_k]

    def query(self, question: str, history: list = None) -> dict:
        """
        Full RAG pipeline with optional multi-turn memory:
          question (+history) → rewrite → LLM keyword expansion → BM25 retrieval → LLM answer

        `history` is a list of {"role": "user"|"assistant", "content": str}, oldest first.
        Pass None or [] for a stateless, single-turn call (backwards compatible).
        """
        try:
            history_text = self._format_history(history or [])

            # Make follow-up questions self-contained before retrieval
            search_question = self._rewrite_question(question, history_text)

            relevant_docs = self._expand_and_retrieve(search_question)
            context = "\n\n---\n\n".join(doc["text"] for doc in relevant_docs)

            answer_prompt = self.answer_prompt.format(
                history=history_text, context=context, question=question
            )
            response = self.llm.invoke(answer_prompt)

            sources = []
            seen = set()
            for doc in relevant_docs:
                key = os.path.basename(doc["source"])
                if key not in seen:
                    seen.add(key)
                    sources.append({"source": doc["source"], "page": doc["page"]})
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
        return list(
            {os.path.basename(doc["source"]) for doc in self.documents}
        )