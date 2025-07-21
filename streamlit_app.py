# import streamlit as st
# from app.retriever import load_pdf_and_create_vectors
# from app.agent import load_agent
# import os

# # Load vector store and agent only once
# @st.cache_resource
# def setup_agent():
#     pdf_path = "data/underwriting.pdf"
#     if not os.path.exists(pdf_path):
#         st.error("PDF not found at `data/underwriting.pdf`")
#         return None
#     load_pdf_and_create_vectors(pdf_path)
#     return load_agent()

# # UI
# st.set_page_config(page_title="Insurance RAG Agent 💡", layout="wide")
# st.title("🧠 Insurance Assistant - Ask your underwriting queries")

# st.markdown("Ask questions based on the underwriting document.")

# query = st.text_input("📩 Enter your question:", placeholder="e.g. What are the claim conditions for critical illness?")
# submit = st.button("🔍 Get Answer")

# if query and submit:
#     with st.spinner("Thinking... 🤖"):
#         agent = setup_agent()
#         if agent:
#             try:
#                 response = agent.invoke({"query": query})
#                 st.success("✅ Answer:")
#                 st.write(response["result"])

#                 with st.expander("📚 Source Documents"):
#                     for i, doc in enumerate(response["source_documents"]):
#                         st.markdown(f"**Document {i+1}:**")
#                         st.write(doc.page_content)
#             except Exception as e:
#                 st.error(f"❌ Error: {e}")

import streamlit as st
from app.retriever import load_pdf_and_create_vectors
from app.agent import load_agent
import os

st.set_page_config(page_title="Insurance AI Agent", layout="wide")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Insurance RAG Assistant")
st.markdown("Ask questions based on your underwriting PDF.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_pdf = st.file_uploader("Upload Underwriting PDF", type="pdf")
    
    model_name = st.selectbox("Select Model", [
        "llama3-70b-8192",
        "gemma-7b-it"
    ])

    if st.button("🔄 Load Agent"):
        
        if uploaded_pdf:
            # Ensure 'data/' folder exists
            os.makedirs("data", exist_ok=True)
            # Save the uploaded PDF
            pdf_path = os.path.join("data", uploaded_pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getvalue())

    with st.spinner("Processing PDF and creating vectorstore..."):
        load_pdf_and_create_vectors(pdf_path)
        st.success("✅ PDF processed and vectorstore created!")


            with st.spinner("Processing PDF and creating vectorstore..."):
                load_pdf_and_create_vectors(pdf_path)
                st.session_state.agent = load_agent(model_name=model_name)

            st.success("Agent ready! Start asking questions.")
        else:
            st.warning("📄 Please upload a PDF first.")

# --- Main Chat Interface ---
if "agent" in st.session_state:
    user_query = st.text_input("💬 Ask a question")

    if user_query:
        with st.spinner("Generating response..."):
            response = st.session_state.agent.invoke({"query": user_query})
            answer = response["result"]

        # Save to history
        st.session_state.messages.append(("user", user_query))
        st.session_state.messages.append(("bot", answer))

    # Show chat history
    for role, msg in reversed(st.session_state.messages):
        if role == "user":
            st.markdown(f"🧑‍💼 **You**: {msg}")
        else:
            st.markdown(f"🤖 **Agent**: {msg}")
else:
    st.info("👈 Please upload a PDF and load the agent to begin.")
