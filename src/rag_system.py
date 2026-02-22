import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()

class RAGSystem:
    def __init__(self, persist_dir: str = "chroma_db", docs_path: str = "documents"):
        # Initialize the system, set paths, and environment variables
        self.persist_dir = persist_dir
        self.docs_path = docs_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to your environment.")

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.1,
        )

        self.vectorstore = None
        self.qa_chain = None
        self._load_or_build_vectorstore()
        self._create_qa_chain()
        
        import chromadb
        chromadb.Client().set_telemetry_enabled(False)

    def _load_or_build_vectorstore(self):
        """Load the vectorstore if it exists, or build it from documents if it doesn't."""
        # Check if the chroma_db directory exists and has data
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                )
                # Verify it has documents (if empty, rebuild it)
                collection = self.vectorstore.get()
                if collection and len(collection.get("ids", [])) > 0:
                    print(f"✅ Vector database loaded from '{self.persist_dir}' with {len(collection['ids'])} chunks")
                    return
                else:
                    print(f"⚠️ Vector database at '{self.persist_dir}' is empty, rebuilding...")
            except Exception as e:
                print(f"⚠️ Could not load existing vector database: {e}, rebuilding...")

        # Rebuild the vectorstore from documents if it doesn't exist or is empty
        self._build_vectorstore()

    def _build_vectorstore(self):
        """Build the vectorstore from PDF documents in the documents directory."""
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(
                f"Documents directory '{self.docs_path}' not found. "
                "Make sure your PDF files are in the 'documents/' folder."
            )

        pdf_files = [f for f in os.listdir(self.docs_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in '{self.docs_path}'. "
                "Add your portfolio PDFs to the 'documents/' folder."
            )

        print(f"📄 Building vector database from {len(pdf_files)} PDF(s): {pdf_files}")

        # Load documents using DirectoryLoader and PyPDFLoader
        loader = DirectoryLoader(self.docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"📄 Loaded {len(documents)} pages from PDFs")

        # Split the documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        print(f"✂️ Split into {len(chunks)} chunks")

        # Build the vectorstore and persist it to the disk
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        print(f"✅ Vector database built and persisted to '{self.persist_dir}'")

    def _create_qa_chain(self):
        """Create the QA chain using the vectorstore."""
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
        """Query the system with a user question."""
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
        """Return a list of available documents in the vector store."""
        try:
            docs = self.vectorstore.get()
            return list({os.path.basename(md["source"]) for md in docs.get("metadatas", [])})
        except Exception as e:
            print(f"Error getting available documents: {e}")
            return []