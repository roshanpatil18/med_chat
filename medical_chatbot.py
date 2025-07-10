#!/usr/bin/env python3
import os
import re
import streamlit as st
from dotenv import load_dotenv
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Configuration
PDF_PATH = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
INDEX_DIR = "faiss_index"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"  # More reliable free model

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
        chunk_size=600,  # Smaller chunks
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "! ", "? ", "] ", ") "]  # Split at natural breaks
    )
    return splitter.split_documents(documents)

def get_vector_store():
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embedder, allow_dangerous_deserialization=True)
    else:
        chunks = load_and_process_pdf()
        db = FAISS.from_documents(chunks, embedder)
        db.save_local(INDEX_DIR)
        st.success("✅ Medical knowledge base ready!")
        return db

def clean_context(context):
    """Remove hyphens at end of lines and join words"""
    context = re.sub(r'-\s+', '', context)  # Remove hyphen line breaks
    context = re.sub(r'\s+', ' ', context)  # Replace multiple spaces
    return context.strip()

def query_huggingface(context, question):
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Clean and format context
    context = clean_context(context)
    
    # Create concise prompt
    prompt = f"""
    Answer this medical question using ONLY the context below.
    If the answer isn't in the context, say "I don't know".
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 500,
            "temperature": 0.1,  # Lower for more factual responses
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        answer = response.json()[0]['generated_text'].strip()
        
        # Ensure answer completes sentences
        if answer and answer[-1] not in {'.', '!', '?'}:
            last_period = answer.rfind('.')
            if last_period != -1:
                answer = answer[:last_period+1]
        return answer
    except Exception as e:
        st.error(f"⚠️ Model unavailable: Using document content directly")
        # Return complete sentences from context
        sentences = re.split(r'(?<=[.!?])\s+', context)
        return ". ".join(sentences[:3]) + "..."

# Streamlit UI setup
st.set_page_config(
    page_title="Medical Encyclopedia Chatbot",
    page_icon="🩺",
    layout="centered"
)

# Sidebar for settings
with st.sidebar:
    st.header("Medical Chatbot Settings")
    st.markdown("""
    This chatbot answers questions using **The Gale Encyclopedia of Medicine**.
    All answers come directly from the PDF content.
    """)
    
    if st.button("Initialize Knowledge Base"):
        with st.spinner("Building medical knowledge base..."):
            st.session_state.db = get_vector_store()

# Main content area
st.title("🩺 Medical Encyclopedia Chatbot")
st.caption("Ask questions about medical conditions, treatments, and first aid procedures")

# Initialize vector store if not loaded
if st.session_state.db is None:
    st.session_state.db = get_vector_store()

# Chat history display
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry['question'])
    with st.chat_message("assistant"):
        st.write(entry['answer'])
        if 'sources' in entry:
            with st.expander("Source Passages"):
                for i, source in enumerate(entry['sources']):
                    st.markdown(f"**Passage {i+1}**")
                    st.write(source)
                    st.divider()

# Question input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user question to history
    st.session_state.history.append({"question": prompt, "answer": ""})
    
    # Display user question
    with st.chat_message("user"):
        st.write(prompt)
    
    # Retrieve relevant context (more chunks)
    docs = st.session_state.db.similarity_search(prompt, k=5)
    context = "\n".join([doc.page_content for doc in docs])
    source_texts = [clean_context(doc.page_content) for doc in docs]
    
    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Consulting medical encyclopedia..."):
            answer = query_huggingface(context, prompt)
            # Update history
            st.session_state.history[-1]['answer'] = answer
            st.session_state.history[-1]['sources'] = source_texts
            st.write(answer)
    
    # Show context sources
    with st.expander("View source passages"):
        for i, source in enumerate(source_texts):
            st.markdown(f"**Passage {i+1}**")
            st.write(source)
            st.divider()