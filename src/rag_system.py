import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()

class RAGSystem:
    def __init__(self, persist_dir: str = "chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
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
            You are an AI assistant helping visitors explore Shubham Jagtap's professional background, AI/ML projects, and research contributions.

Use only the provided context below to answer the user's question. If the answer is not in the context, politely say you don't know.

Context:
{context}

User Question:
{question}

Instructions:
- If the user says hello or greets you, reply warmly and offer to help.
- If the user's question is unclear or too short, ask them to clarify.
- Otherwise, give a detailed, specific, and helpful answer using the context above.
- Always cite the source document (e.g., [source_name.pdf, page 2]).

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
