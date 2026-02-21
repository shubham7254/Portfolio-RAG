import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

class RAGSystem:
    def __init__(self, persist_dir: str = "chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.1,
        )
        self.vectorstore = None
        self.qa_chain = None
        self._load_vectorstore()
        self._create_qa_chain()

    def _load_vectorstore(self):
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            print(f"✅ Vector database loaded from '{self.persist_dir}'")
        except Exception as e:
            print(f"❌ Error loading vector database: {e}")
            raise

    def _create_qa_chain(self):
        template = """
        You are an intelligent and professional AI assistant designed to help users explore Shubham Jagtap's portfolio. Your role is to interpret user questions and generate clear, specific, and insightful answers based entirely on the context provided.

Your audience includes recruiters, collaborators, and curious professionals.

Your goal is to make the user feel:
- Understood (you correctly grasp their intent)
- Informed (you give factual, relevant, and non-generic answers)
- Engaged (you write in a helpful, confident, but human tone)

---

Guidelines for responding:

1. Read the user's question carefully. Determine whether it is:
   - A general greeting (e.g., “hello”, “hi”)
   - A vague or unclear query (e.g., “okay?”, “?”)
   - A specific question (e.g., “Tell me about model training experience”)

2. Choose your response strategy:
   - If it's a greeting, respond briefly, warmly, and invite them to ask a question.
   - If it's unclear, politely ask for clarification.
   - If it's specific, give a detailed, factual answer based on the context.

3. When answering specific questions:
   - Mention project names, datasets, or tools wherever relevant.
   - Highlight outcomes, metrics, or evaluations that show impact.
   - Use the user's language when possible (e.g., if they ask about “LLMs”, respond using that term).
   - If the answer is not in the context, say so clearly and invite the user to ask a different question.

4. Always maintain a clear and professional tone:
   - Friendly but not overly casual
   - Confident but never speculative
   - Avoid robotic phrasing or repeating question words

5. Cite sources at the end of your answer like this:
   (Source: [filename.pdf, Section 2.3]) or (Source: [Context, Project: Legal Summarizer])

---

Context:
{context}

User Question:
{question}

Answer:
"""

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    def query(self, question: str) -> dict:
        try:
            response = self.qa_chain.invoke({"query": question})
            sources = [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                }
                for doc in response.get("source_documents", [])
            ]
            return {"answer": response["result"], "sources": sources, "status": "success"}
        except Exception as e:
            return {"answer": f"Error: {e}", "sources": [], "status": "error"}

    def get_available_documents(self) -> list[str]:
        try:
            docs = self.vectorstore.get()
            return list({os.path.basename(md["source"]) for md in docs.get("metadatas", [])})
        except Exception as e:
            print(f"Error getting available documents: {e}")
            return []
