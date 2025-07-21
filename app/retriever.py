import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_pdf_and_create_vectors(pdf_path, vector_store_path="vectorstore"):
    print("📄 Reading PDF and generating chunks...")
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])

    # Chunking the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = splitter.create_documents([text])

    # Using free embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local(vector_store_path)
    print("✅ Vector DB saved to:", vector_store_path)
    return vector_db
