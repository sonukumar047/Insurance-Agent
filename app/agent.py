import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

load_dotenv()

def load_agent(vector_store_path="vectorstore", model_name="llama3-70b-8192"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_name
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
