import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Free alternative
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv('secrets.txt')

class DocumentProcessor:
    def __init__(
        self,
        docs_path: str = "documents",
        persist_dir: str = "chroma_db"
    ):
        self.docs_path = docs_path
        self.persist_dir = persist_dir
        # Use free HuggingFace embeddings instead of OpenAI
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    def load_documents(self):
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        return loader.load()

    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_documents(documents)

    def create_vectorstore(self):
        docs = self.load_documents()
        chunks = self.split_documents(docs)
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        vs.persist()
        return vs

if __name__ == "__main__":
    processor = DocumentProcessor()
    vectorstore = processor.create_vectorstore()
    print(f"✅ Done! Vector store persisted to '{processor.persist_dir}'")
