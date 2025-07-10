#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import traceback

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("Missing HUGGINGFACEHUB_API_TOKEN in .env file")

# Config
PDF_PATH = r"C:\Users\rosha\OneDrive\Documents\chatbot\medical-chatbot\data\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
INDEX_DIR = "faiss_index"
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

def load_and_split(path: str):
    """Load and split PDF into chunks."""
    loader = UnstructuredPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def build_or_load_index(docs, embedder):
    """Load existing FAISS index or create a new one."""
    if os.path.isdir(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embedder, allow_dangerous_deserialization=True)
    vs = FAISS.from_documents(docs, embedder)
    vs.save_local(INDEX_DIR)
    return vs

def ask_llm(query: str, retriever, llm) -> str:
    """Query the LLM with RAG."""
    try:
        hits = retriever.invoke(query)  # Simplified invocation
        if not hits:
            return "⚠️ No relevant context found."
        
        context = "\n\n".join(d.page_content for d in hits)
        prompt = f"""Answer the question based on this context:
{context}

Question: {query}
Answer:"""
        
        response = llm(prompt)  # Direct invocation for HuggingFaceEndpoint
        return str(response).strip()
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        traceback.print_exc()
        return "❌ Failed to generate answer."

def main():
    # Validate PDF exists
    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

    print("🔄 Loading & splitting PDF...")
    docs = load_and_split(PDF_PATH)

    print("🔄 Setting up embeddings...")
    embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("🔄 Creating/loading FAISS index...")
    vs = build_or_load_index(docs, embedder)
    retriever = vs.as_retriever(search_kwargs={"k": 3})  # Return top 3 hits

    print("🔄 Initializing LLM...")
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.7,
        max_new_tokens=512,  # Increased for better answers
    )

    print("✅ Ready! Ask your medical questions (type 'exit' to quit):")
    while True:
        try:
            query = input("\nQuery> ").strip()
            if query.lower() in ("exit", "quit"):
                break
            
            print("\n🔍 Searching...")
            answer = ask_llm(query, retriever, llm)
            print(f"\n📘 Answer:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    main()