#!/usr/bin/env python3
import os
import re
import streamlit as st
from dotenv import load_dotenv
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
# Fixed import - use langchain_huggingface instead of langchain.embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# Configuration
PDF_PATH = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
INDEX_DIR = "faiss_index"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
if 'history' not in st.session_state:
    st.session_state.history = []

def load_and_process_pdf():
    st.info("📚 Loading and processing medical encyclopedia...")
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "! ", "? ", "] ", ") "]
    )
    return splitter.split_documents(documents)

def get_vector_store():
    try:
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=".hf_cache"
        )
    except ImportError as e:
        st.error(f"Error initializing embeddings: {e}")
        st.error("Please install required dependencies: pip install sentence-transformers")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        st.stop()

    if os.path.isdir(INDEX_DIR):
        try:
            return FAISS.load_local(INDEX_DIR, embedder, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Could not load existing index: {e}")
            st.info("Creating new index...")
            # If loading fails, create new index
            chunks = load_and_process_pdf()
            db = FAISS.from_documents(chunks, embedder)
            db.save_local(INDEX_DIR)
            st.success("✅ Knowledge base ready!")
            return db
    else:
        chunks = load_and_process_pdf()
        db = FAISS.from_documents(chunks, embedder)
        db.save_local(INDEX_DIR)
        st.success("✅ Knowledge base ready!")
        return db

def clean_context(text: str) -> str:
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def query_model(context: str, question: str) -> str:
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    prompt = (
        "Answer this medical question using ONLY the context below. "
        "If the answer isn't in the context, say \"I don't know.\""
        "\n\nCONTEXT:\n" + clean_context(context) +
        "\n\nQUESTION:\n" + question +
        "\n\nANSWER:\n"
    )

    payload = {
        "inputs": prompt,
        "parameters": { "max_length": 512, "temperature": 0.0 }
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        gen = r.json()
        return gen[0]["generated_text"].strip()
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', context)
        return " ".join(sentences[:2]) + "..."

# --- Streamlit App Layout ---
st.set_page_config(page_title="Medical Encyclopedia Chatbot", page_icon="🩺")
st.title("🩺 Medical Encyclopedia Chatbot")
st.caption("Powered by The Gale Encyclopedia of Medicine")

with st.sidebar:
    st.header("Settings")
    st.write("This bot answers from a medical PDF.")
    if st.button("(Re)build knowledge base"):
        with st.spinner("Indexing..."):
            st.session_state.db = get_vector_store()

# Initialize the database with error handling
try:
    if st.session_state.db is None:
        st.session_state.db = get_vector_store()
except Exception as e:
    st.error(f"Failed to initialize knowledge base: {e}")
    st.stop()

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry.get("sources"):
            with st.expander("Source Passages"):
                for s in entry["sources"]:
                    st.write(s)
                st.divider()

if user_q := st.chat_input("Ask a medical question..."):
    st.session_state.history.append({"question": user_q, "answer": ""})

    with st.chat_message("user"):
        st.write(user_q)

    docs = st.session_state.db.similarity_search(user_q, k=5)
    ctx = "\n".join([d.page_content for d in docs])
    sources = [clean_context(d.page_content) for d in docs]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans = query_model(ctx, user_q)
            st.session_state.history[-1].update(answer=ans, sources=sources)
            st.write(ans)

    with st.expander("View source passages"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**Passage {i}:**")
            st.write(s)
            st.divider()