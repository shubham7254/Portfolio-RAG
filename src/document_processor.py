import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

class DocumentProcessor:
    def __init__(self, docs_path: str = "documents", persist_dir: str = "chroma_db"):
        self.docs_path = docs_path
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def load_documents(self):
        loader = DirectoryLoader(self.docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        return loader.load()

    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents)

    def create_vectorstore(self):
        docs = self.load_documents()
        chunks = self.split_documents(docs)
        vs = Chroma.from_documents(
            documents=chunks, embedding=self.embeddings, persist_directory=self.persist_dir
        )
        return vs

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.create_vectorstore()
    print(f"✅ Vector store persisted to '{processor.persist_dir}'")
