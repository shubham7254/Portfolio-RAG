import os
from typing import List
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

# Load API keys
load_dotenv('secrets.txt')
load_dotenv()  # fallback to .env


class FilteredRetriever(BaseRetriever):
    """Wraps a base retriever and removes code-heavy chunks before they reach the LLM."""
    base_retriever: object

    def _is_code_heavy(self, text: str) -> bool:
        # Check character density
        code_chars = sum(1 for c in text if c in '(){}[]<>=;:/\\|#@$%^&*')
        if (code_chars / max(len(text), 1)) > 0.10:
            return True
        # Check for code keywords
        code_keywords = ['def ', 'import ', 'return ', 'sorted(', 'lambda ', 'list(', '.keys()', 'for ', 'if __']
        matches = sum(1 for kw in code_keywords if kw in text)
        return matches >= 2

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        filtered = [doc for doc in docs if not self._is_code_heavy(doc.page_content)]
        # Fall back to all docs if everything got filtered out
        return filtered if filtered else docs


class RAGSystem:
    def __init__(self, persist_dir: str = "chroma_db", docs_path: str = "documents"):
        self.persist_dir = persist_dir
        self.docs_path = docs_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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

        self.vectorstore = None
        self.qa_chain = None
        self._load_or_build_vectorstore()
        self._create_qa_chain()

    def _load_or_build_vectorstore(self):
        """Load the vectorstore if it exists, or build it from documents if it doesn't."""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                )
                collection = self.vectorstore.get()
                if collection and len(collection.get("ids", [])) > 0:
                    print(f"✅ Vector database loaded from '{self.persist_dir}' with {len(collection['ids'])} chunks")
                    return
                else:
                    print(f"⚠️ Vector database at '{self.persist_dir}' is empty, rebuilding...")
            except Exception as e:
                print(f"⚠️ Could not load existing vector database: {e}, rebuilding...")

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

        loader = DirectoryLoader(self.docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"📄 Loaded {len(documents)} pages from PDFs")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        chunks = splitter.split_documents(documents)
        print(f"✂️ Split into {len(chunks)} chunks")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        print(f"✅ Vector database built and persisted to '{self.persist_dir}'")

    def _create_qa_chain(self):
        """Create the question-answering chain with custom prompt"""

        template = """You are a smart AI assistant on a personal portfolio website. \
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

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        base_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20}
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=FilteredRetriever(base_retriever=base_retriever),
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
            return {"answer": f"I apologize, but I encountered an error: {str(e)}", "sources": [], "status": "error"}

    def get_available_documents(self) -> list:
        """Return a list of available documents in the vector store."""
        try:
            docs = self.vectorstore.get()
            return list({os.path.basename(md["source"]) for md in docs.get("metadatas", []) if "source" in md})
        except Exception as e:
            print(f"Error getting available documents: {e}")
            return []
