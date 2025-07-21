# 🛡️ Insurance RAG Agent with Groq + Streamlit

A Retrieval-Augmented Generation (RAG) chatbot designed for the **insurance domain** that uses underwriting PDFs and answers queries intelligently using **Groq LLMs** and **FAISS vector store**.

Built with:
- 🔗 [LangChain](https://www.langchain.com/)
- 🤗 [HuggingFace Embeddings](https://huggingface.co/)
- 🧠 [Groq LLMs](https://console.groq.com/)
- 🌐 [Streamlit](https://streamlit.io/) for UI

---

## 📂 Project Structure

```bash
Insurance-Agent/
│
├── app/
│   ├── agent.py             # Loads Groq LLM and builds QA chain
│   ├── retriever.py         # PDF loader and vector DB builder
│
├── data/                    # Place PDF files here
│
├── vectorstore/             # FAISS vector DB (auto-generated)
│
├── .env                     # Store your GROQ_API_KEY
├── main.py                  # CLI version of the app
├── ui_streamlit.py          # 📱 Streamlit web UI
├── requirements.txt         # Python dependencies
└── README.md
```


## 🚀 Features
✅ Upload insurance underwriting PDFs
✅ RAG using FAISS + HuggingFace Embeddings
✅ Uses blazing fast Groq LLMs (LLaMA3, Gemma, Mixtral*)
✅ Clean and interactive Streamlit UI
✅ CLI and Web versions available
✅ Returns both answers and source references

## 🔧 Setup Instructions
1. Clone the Repository

git clone https://github.com/yourusername/insurance-rag-agent.git
cd insurance-rag-agent

 Create and Activate a Virtual Environment

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install Python Dependencies
pip install -r requirements.txt

4. Create .env File
# .env
GROQ_API_KEY=your_groq_api_key

▶️ Run the Application
✅ Option 1: Use Streamlit Web App

streamlit run ui_streamlit.py

💻 Option 2: Use Command-Line Interface (CLI)
python main.py

##📦 Requirements
streamlit
langchain
langchain-community
langchain-groq
huggingface_hub[hf_xet]
faiss-cpu
python-dotenv

