import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API keys
load_dotenv('secrets.txt')

class RAGSystem:
    def __init__(self, persist_dir: str = "chroma_db"):
        self.persist_dir = persist_dir
        
        # Initialize the same embedding model used for document processing
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize Groq LLM with your API key
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",  # Fast and high-quality model
            temperature=0.1  # Low temperature for more factual responses
        )
        
        # Load the existing vector database
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
        
        # Custom prompt template for portfolio queries
        template = """You are an AI assistant helping visitors learn about this person's professional background, projects, and research.

Use the following context from their resume, project reports, and research papers to answer questions accurately and professionally.

Context: {context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- Be specific and detailed when describing projects, skills, or experience
- If you mention technical details, include relevant specifics from the documents
- Always cite which document(s) your information comes from
- If the context doesn't contain enough information to answer the question, say so politely
- Use a professional, informative tone suitable for a portfolio website

Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            ),
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
            # Get response from the QA chain
            response = self.qa_chain.invoke({"query": question})
            
            # Extract source documents
            sources = []
            for doc in response.get("source_documents", []):
                source_info = {
                    "content": doc.page_content[:200] + "...",  # First 200 chars
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
            # Get a sample of documents to see what's available
            docs = self.vectorstore.get()
            sources = set()
            for metadata in docs.get("metadatas", []):
                if "source" in metadata:
                    # Extract just the filename from the full path
                    source_file = os.path.basename(metadata["source"])
                    sources.add(source_file)
            
            return list(sources)
        except Exception as e:
            print(f"Error getting available documents: {e}")
            return []

# Test function to verify the system works
def test_rag_system():
    """Test the RAG system with sample questions"""
    
    print("🔧 Testing RAG System...")
    rag = RAGSystem()
    
    # Test questions
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
            for j, source in enumerate(result['sources'][:2], 1):  # Show first 2 sources
                print(f"  {j}. {source['source']} (page {source['page']})")
        else:
            print(f"Error: {result['answer']}")
        
        print()

if __name__ == "__main__":
    test_rag_system()
