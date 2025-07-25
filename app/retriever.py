import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf_and_create_vectors(pdf_paths, vector_store_path="vectorstore"):
    """
    Load multiple PDF files and create a vectorstore with enhanced error handling
    
    Args:
        pdf_paths: Can be a single path (string) or list of paths (list of strings)
        vector_store_path: Path where to save the vector store
    """
    print(f"🔍 Function called with: {pdf_paths}")
    print(f"🔍 Type: {type(pdf_paths)}")
    print(f"🔍 Current working directory: {os.getcwd()}")
    
    # Normalize to list of strings
    if isinstance(pdf_paths, str):
        pdf_paths = [os.path.normpath(pdf_paths)]
    elif isinstance(pdf_paths, list):
        pdf_paths = [os.path.normpath(p) for p in pdf_paths if isinstance(p, str)]
    else:
        raise ValueError(f"Invalid input type for pdf_paths: {type(pdf_paths)}. Must be str or list of str.")

    if not pdf_paths:
        raise ValueError("No valid PDF paths provided.")

    print(f"🔍 Normalized paths: {pdf_paths}")
    
    # Check file system state
    if os.path.exists('data'):
        print(f"🔍 Data directory contents: {os.listdir('data')}")
    else:
        print("🔍 Data directory does not exist")

    # Load documents from all PDFs
    all_documents = []
    processed_files = []
    
    for pdf_path in pdf_paths:
        if not isinstance(pdf_path, str):
            print(f"❌ Invalid path type: {type(pdf_path)}")
            continue
        
        print(f"🔍 Checking path: {pdf_path}")
        print(f"🔍 Path exists: {os.path.exists(pdf_path)}")
        
        if os.path.exists(pdf_path):
            try:
                # Check file size
                file_size = os.path.getsize(pdf_path)
                print(f"🔍 File size: {file_size} bytes")
                
                if file_size == 0:
                    print(f"❌ File is empty: {pdf_path}")
                    continue
                
                # Try to load the PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                if not documents:
                    print(f"❌ No documents loaded from: {pdf_path}")
                    continue
                
                # Add source information to metadata
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(pdf_path)
                
                all_documents.extend(documents)
                processed_files.append(pdf_path)
                print(f"✅ Successfully loaded {len(documents)} pages from: {pdf_path}")
                
            except Exception as e:
                print(f"❌ Error loading PDF {pdf_path}: {str(e)}")
                continue
        else:
            print(f"❌ Warning: PDF file not found: {pdf_path}")
            # List what's actually in the directory
            dir_path = os.path.dirname(pdf_path) or '.'
            if os.path.exists(dir_path):
                print(f"🔍 Contents of {dir_path}: {os.listdir(dir_path)}")
    
    print(f"🔍 Total documents loaded: {len(all_documents)} from {len(processed_files)} files")
    
    if not all_documents:
        error_msg = f"No valid PDF documents found to process. Checked {len(pdf_paths)} paths: {pdf_paths}"
        print(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"🔍 Created {len(chunks)} chunks")
    
    # Create embeddings
    print("🔄 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create and save vectorstore
    print("🔄 Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vector_store_path)
    
    print(f"✅ Vectorstore created successfully with {len(chunks)} chunks from {len(processed_files)} PDF(s)")
    return vectorstore
