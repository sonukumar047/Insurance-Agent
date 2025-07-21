import streamlit as st
from app.retriever import load_pdf_and_create_vectors
from app.agent import load_agent
import os

# Page config
st.set_page_config(page_title="Smart AI Agent", layout="wide")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Smart RAG Assistant")
st.markdown("Ask questions based on PDF document.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_pdf = st.file_uploader("Upload your  PDF document", type="pdf")

    model_name = st.selectbox("Select Model", [
        "llama3-70b-8192",
        "gemma2-9b-it",
        "qwen/qwen3-32b",
        "deepseek-r1-distill-llama-70b",
        "compound-beta"
    ])

    if st.button("🔄 Load Agent"):
        if uploaded_pdf:
            # Ensure the 'data/' folder exists
            os.makedirs("data", exist_ok=True)

            # Save uploaded PDF to 'data/' directory
            pdf_path = os.path.join("data", uploaded_pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getvalue())

            st.success("✅ PDF uploaded and saved successfully!")

            with st.spinner("Processing PDF and creating vectorstore..."):
                load_pdf_and_create_vectors(pdf_path)
                st.success("✅ PDF processed and vectorstore created!")

            with st.spinner("Loading agent..."):
                st.session_state.agent = load_agent(model_name=model_name)
                st.success("🎉 Agent is ready! Start asking your questions.")
        else:
            st.warning("⚠️ Please upload a PDF first.")

# --- Main Chat Interface ---
if "agent" in st.session_state:
    user_query = st.text_input("💬 Ask a question")

    if user_query:
        with st.spinner("Generating response..."):
            try:
                response = st.session_state.agent.invoke({"query": user_query})
                answer = response["result"]

                # Save to chat history
                st.session_state.messages.append(("user", user_query))
                st.session_state.messages.append(("bot", answer))
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Display chat history
    for role, msg in reversed(st.session_state.messages):
        if role == "user":
            st.markdown(f"🧑‍💼 **You**: {msg}")
        else:
            st.markdown(f"🤖 **Agent**: {msg}")
else:
    st.info("👈 Please upload a PDF and click 'Load Agent' to start.")
