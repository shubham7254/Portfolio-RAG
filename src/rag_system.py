import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from dotenv import load_dotenv

# Load API keys
load_dotenv('secrets.txt')


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
    def __init__(self, persist_dir: str = "chroma_db"):
        self.persist_dir = persist_dir

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )

        self.vectorstore = None
        self.qa_chain = None
        self._load_vectorstore()
        self._create_qa_chain()

    def _load_vectorstore(self):
        """Load the existing Chroma vector database"""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            print(f"✅ Vector database loaded from '{self.persist_dir}'")
        except Exception as e:
            print(f"❌ Error loading vector database: {e}")
            raise

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
- Answer in a natural, confident, first-person-about-them tone (e.g. "He has..." or "They have...")
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
            return_source_documents=True
        )

    def query(self, question: str):
        """
        Process a question and return an answer with sources

        Args:
            question (str): The user's question

        Returns:
            dict: Contains 'answer' and 'sources'
        """
        try:
            response = self.qa_chain.invoke({"query": question})

            sources = []
            for doc in response.get("source_documents", []):
                source_info = {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A")
                }
                sources.append(source_info)

            return {
                "answer": response["result"],
                "sources": sources,
                "status": "success"
            }

        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "status": "error"
            }

    def get_available_documents(self):
        """Return a list of documents available in the vector database"""
        try:
            docs = self.vectorstore.get()
            sources = set()
            for metadata in docs.get("metadatas", []):
                if "source" in metadata:
                    source_file = os.path.basename(metadata["source"])
                    sources.add(source_file)
            return list(sources)
        except Exception as e:
            print(f"Error getting available documents: {e}")
            return []


def test_rag_system():
    """Test the RAG system with sample questions"""

    print("🔧 Testing RAG System...")
    rag = RAGSystem()

    test_questions = [
        "What programming languages and technical skills does this person have?",
        "Tell me about their machine learning or AI projects.",
        "What is their educational background?",
        "What research has this person conducted?"
    ]

    print(f"\n📄 Available documents: {rag.get_available_documents()}")

    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test Question {i}: {question}")
        print("-" * 50)

        result = rag.query(question)

        if result["status"] == "success":
            print(f"Answer: {result['answer']}")
            print(f"\nSources: {len(result['sources'])} document(s)")
            for j, source in enumerate(result['sources'][:2], 1):
                print(f"  {j}. {source['source']} (page {source['page']})")
        else:
            print(f"Error: {result['answer']}")

        print()


if __name__ == "__main__":
    test_rag_system()
