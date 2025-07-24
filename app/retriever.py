import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf_and_create_vectors(pdf_paths):
    """
    Load multiple PDF files and create a vectorstore
    
    Args:
        pdf_paths: Can be a single path (string) or list of paths (list of strings)
    """
    # Normalize to list of strings
    if isinstance(pdf_paths, str):
        pdf_paths = [os.path.normpath(pdf_paths)]  # Normalize path for OS compatibility
    elif isinstance(pdf_paths, list):
        pdf_paths = [os.path.normpath(p) for p in pdf_paths if isinstance(p, str)]
    else:
        raise ValueError(f"Invalid input type for pdf_paths: {type(pdf_paths)}. Must be str or list of str.")

    if not pdf_paths:
        raise ValueError("No valid PDF paths provided.")

    # Load documents from all PDFs
    all_documents = []
    
    for pdf_path in pdf_paths:
        if not isinstance(pdf_path, str):
            raise ValueError(f"Invalid path type: {type(pdf_path)}. Each path must be a string.")
        
        if os.path.exists(pdf_path):
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Add source information to metadata
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(pdf_path)
                
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading PDF {pdf_path}: {str(e)}")
        else:
            print(f"Warning: PDF file not found: {pdf_path}")
    
    if not all_documents:
        raise ValueError("No valid PDF documents found to process")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create and save vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    
    print(f"Vectorstore created successfully with {len(chunks)} chunks from {len(pdf_paths)} PDF(s)")
    return vectorstore
