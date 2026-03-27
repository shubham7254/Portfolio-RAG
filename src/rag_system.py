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

        # Separate resume chunks (always included) from the rest
        self.resume_chunks = [
            doc for doc in self.documents if "Resume" in doc["source"]
        ]
        other_chunks = [
            doc for doc in self.documents if "Resume" not in doc["source"]
        ]

        # BM25 index over non-resume documents only
        tokenized_corpus = [doc["text"].lower().split() for doc in other_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.other_chunks = other_chunks

        print(
            f"✅ Loaded {len(self.documents)} chunks "
            f"({len(self.resume_chunks)} resume + {len(other_chunks)} project/research)"
        )

        # Prompt 1: extract search keywords from the question (tiny call ~100 tokens)
        self.keyword_prompt = PromptTemplate(
            template="""Extract 6-10 specific search keywords from this question that would \
help find relevant information in project reports and research papers.
Return ONLY the keywords separated by spaces, nothing else.

Question: {question}
Keywords:""",
            input_variables=["question"],
        )

        # Prompt 2: answer using full profile context
        self.answer_prompt = PromptTemplate(
            template="""You are a smart AI assistant on Shubham Jagtap's portfolio website. \
Visitors ask about his background, skills, projects, and experience.

Use the context below to answer the visitor's question. The context includes his \
full resume plus relevant project/research excerpts.

Context:
{context}

Visitor's question: {question}

Guidelines:
- Always answer based on what IS in the context — never say "not mentioned" or "not found"
- If someone asks about experience, calculate or infer from education dates, projects, \
  and skills (e.g. "He has been building AI projects since X, currently pursuing his MS...")
- Answer in a natural, confident tone in third person ("He has...", "Shubham...")
- Synthesize into a coherent answer — don't just list raw facts
- Do NOT mention document names, file names, or where the info came from
- Keep the answer conversational and concise — suitable for a portfolio chat widget

Answer:""",
            input_variables=["context", "question"],
        )

    def _retrieve(self, question: str, k: int = 5):
        """
        Retrieval strategy:
        - Always include all resume chunks (3 chunks, full profile context)
        - LLM expands question → BM25 finds top-k relevant project/research chunks
        - Combined context stays well under 12K TPM limit
        """
        # Step 1: LLM keyword expansion for BM25
        keyword_prompt = self.keyword_prompt.format(question=question)
        keywords_response = self.llm.invoke(keyword_prompt)
        expanded_query = keywords_response.content.strip()

        # Step 2: BM25 over project/research docs
        tokenized_query = expanded_query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        project_docs = [self.other_chunks[i] for i in top_k]

        # Resume always first so LLM sees full profile before project details
        return self.resume_chunks + project_docs

    def query(self, question: str) -> dict:
        """
        RAG pipeline:
          question
            → LLM keyword expansion (~100 tokens)
            → resume chunks (always) + BM25 top-5 project chunks
            → LLM generates answer (~3-4K tokens total)
        """
        try:
            relevant_docs = self._retrieve(question)
            context = "\n\n---\n\n".join(doc["text"] for doc in relevant_docs)
            answer_prompt = self.answer_prompt.format(
                context=context, question=question
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
