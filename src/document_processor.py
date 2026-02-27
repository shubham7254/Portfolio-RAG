import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv('secrets.txt')
load_dotenv()

class DocumentProcessor:
    def __init__(
        self,
        docs_path: str = "documents",
        persist_dir: str = "chroma_db"
    ):
        self.docs_path = docs_path
        self.persist_dir = persist_dir
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
            chunk_size=1500,
            chunk_overlap=300
        )
        return splitter.split_documents(documents)

    def create_vectorstore(self):
        docs = self.load_documents()
        chunks = self.split_documents(docs)
        print(f"📄 Loaded {len(docs)} pages, split into {len(chunks)} chunks")
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        return vs

if __name__ == "__main__":
    processor = DocumentProcessor()
    vectorstore = processor.create_vectorstore()
    print(f"✅ Done! Vector store persisted to '{processor.persist_dir}'")
