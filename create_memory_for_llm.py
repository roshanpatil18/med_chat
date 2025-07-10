#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# ─── Imports ────────────────────────────────────────────────────────────────
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint

# ─── Load ENV ─────────────────────────────────────────────────────────────
load_dotenv()  # pip install python-dotenv
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("Missing HUGGINGFACEHUB_API_TOKEN in environment")

# ─── Configuration ────────────────────────────────────────────────────────
PDF_PATH  = r"C:\Users\rosha\OneDrive\Documents\chatbot\medical-chatbot\data\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
INDEX_DIR = "faiss_index"
MODEL_ID  = "tiiuae/falcon-7b-instruct"

if not os.path.isfile(PDF_PATH):
    raise FileNotFoundError(f"Cannot find file: {PDF_PATH}")

# ─── Build / Load Index ───────────────────────────────────────────────────
loader    = PyMuPDFLoader(PDF_PATH)
documents = loader.load()
splitter  = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks    = splitter.split_documents(documents)

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)  # token is read from env

if os.path.isdir(INDEX_DIR):
    db = FAISS.load_local(INDEX_DIR, embeddings=embedder)
else:
    db = FAISS.from_documents(chunks, embedder)
    db.save_local(INDEX_DIR)

# ─── Set Up LLM & QA Chain ────────────────────────────────────────────────
llm = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.7,
    max_new_tokens=256,
    top_k=50,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=False,
)

# ─── Interactive Loop ────────────────────────────────────────────────────
def main_loop():
    print("✅ QA ready! Type questions (exit/quit to stop).")
    while True:
        try:
            query = input("Query> ").strip()
            if query.lower() in ("exit", "quit"):
                print("👋 Goodbye!")
                break
            result = qa_chain.invoke({"query": query})
            print("\nAnswer:\n", result["result"], "\n")
        except KeyboardInterrupt:
            print("\nInterrupted—exiting.")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main_loop()
