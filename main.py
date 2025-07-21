from app.retriever import load_pdf_and_create_vectors
from app.agent import load_agent
import os

def main():
    # Ensure PDF exists
    pdf_path = "data/underwriting.pdf"
    if not os.path.exists(pdf_path):
        print("❌ PDF not found:", pdf_path)
        return

    # Step 1: Generate vector DB
    load_pdf_and_create_vectors(pdf_path)

    # Step 2: Load Agent with RAG
    agent = load_agent()

    print("\n💬 Ask your insurance questions (type 'exit' to quit):")
    while True:
        query = input(">> ")
        if query.lower() in ["exit", "quit"]: break
        response = agent.invoke({"query": query})
        print("\n💡 Answer:", response["result"])


if __name__ == "__main__":
    main()
