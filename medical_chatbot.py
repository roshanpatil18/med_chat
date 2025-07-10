#!/usr/bin/env python3
import os
import re
import streamlit as st
from dotenv import load_dotenv
import requests
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# Configuration
PDF_PATH = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
INDEX_DIR = "simple_index"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
if 'history' not in st.session_state:
    st.session_state.history = []

class SimpleVectorStore:
    def __init__(self, documents=None):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.documents = documents or []
        self.vectors = None
        self.texts = []
        
        if documents:
            self.texts = [doc.page_content for doc in documents]
            self.vectors = self.vectorizer.fit_transform(self.texts)
    
    def similarity_search(self, query, k=5):
        if self.vectors is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return document-like objects
        results = []
        for i in top_k_indices:
            if similarities[i] > 0:  # Only return relevant results
                doc_obj = type('Document', (), {
                    'page_content': self.texts[i],
                    'metadata': {'similarity': similarities[i]}
                })()
                results.append(doc_obj)
        
        return results
    
    def save_local(self, path):
        os.makedirs(path, exist_ok=True)
        data = {
            'texts': self.texts,
            'vectorizer': self.vectorizer,
            'vectors': self.vectors
        }
        with open(os.path.join(path, 'vector_store.pkl'), 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load_local(cls, path):
        with open(os.path.join(path, 'vector_store.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        store = cls()
        store.texts = data['texts']
        store.vectorizer = data['vectorizer']
        store.vectors = data['vectors']
        return store

def load_and_process_pdf():
    st.info("📚 Loading and processing medical encyclopedia...")
    
    if not os.path.exists(PDF_PATH):
        st.error(f"PDF file not found: {PDF_PATH}")
        st.error("Please upload your PDF file to the data/ directory")
        st.stop()
    
    try:
        loader = PyMuPDFLoader(PDF_PATH)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", "] ", ") ", " "]
        )
        chunks = splitter.split_documents(documents)
        
        st.success(f"✅ Loaded {len(chunks)} text chunks from PDF")
        return chunks
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        st.stop()

def get_vector_store():
    if os.path.exists(os.path.join(INDEX_DIR, 'vector_store.pkl')):
        try:
            st.info("Loading existing knowledge base...")
            return SimpleVectorStore.load_local(INDEX_DIR)
        except Exception as e:
            st.warning(f"Could not load existing index: {e}")
            st.info("Creating new index...")
    
    # Create new vector store
    chunks = load_and_process_pdf()
    db = SimpleVectorStore(chunks)
    db.save_local(INDEX_DIR)
    st.success("✅ Knowledge base created successfully!")
    return db

def clean_context(text: str) -> str:
    # Clean up text formatting
    text = re.sub(r'-\s+', '', text)  # Remove hyphenated line breaks
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)  # Remove special chars
    return text.strip()

def query_model(context: str, question: str) -> str:
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    # Create a focused prompt
    prompt = f"""Based on the medical information below, answer the question accurately and concisely.

MEDICAL CONTEXT:
{clean_context(context)}

QUESTION: {question}

ANSWER: """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.1,
            "do_sample": False,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
            # Clean up the response
            if "ANSWER:" in generated_text:
                answer = generated_text.split("ANSWER:")[-1].strip()
            else:
                answer = generated_text.strip()
            
            return answer if answer else "I don't have enough information to answer that question."
        else:
            return "I couldn't generate a response. Please try again."
            
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        # Fallback to context-based answer
        sentences = re.split(r'(?<=[.!?])\s+', context)
        return " ".join(sentences[:3]) + "..."
    except Exception as e:
        st.error(f"Error querying model: {e}")
        return "I encountered an error while processing your question."

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Medical Encyclopedia Chatbot", 
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical Encyclopedia Chatbot")
st.caption("Free medical information chatbot powered by The Gale Encyclopedia of Medicine")

# Sidebar
with st.sidebar:
    st.header("📚 Knowledge Base")
    
    if st.button("🔄 Rebuild Knowledge Base"):
        with st.spinner("Rebuilding knowledge base..."):
            if os.path.exists(INDEX_DIR):
                import shutil
                shutil.rmtree(INDEX_DIR)
            st.session_state.db = get_vector_store()
            st.rerun()
    
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. 📖 Searches medical encyclopedia")
    st.markdown("2. 🔍 Finds relevant passages")
    st.markdown("3. 🤖 Generates answer using AI")
    
    st.markdown("---")
    st.markdown("**Note:** This is for educational purposes only. Always consult healthcare professionals for medical advice.")

# Initialize database
if st.session_state.db is None:
    with st.spinner("Initializing knowledge base..."):
        st.session_state.db = get_vector_store()

# Display chat history
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        
        if entry.get("sources"):
            with st.expander("📚 View Source Passages"):
                for i, source in enumerate(entry["sources"], 1):
                    st.markdown(f"**Passage {i}:**")
                    st.write(source)
                    if i < len(entry["sources"]):
                        st.divider()

# Chat input
if user_question := st.chat_input("Ask a medical question..."):
    # Add to history
    st.session_state.history.append({
        "question": user_question, 
        "answer": "", 
        "sources": []
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_question)
    
    # Process and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching medical encyclopedia..."):
            # Search for relevant documents
            docs = st.session_state.db.similarity_search(user_question, k=5)
            
            if not docs:
                st.write("I couldn't find relevant information in the medical encyclopedia for your question.")
                st.session_state.history[-1]["answer"] = "I couldn't find relevant information in the medical encyclopedia for your question."
            else:
                # Combine context from relevant documents
                context = "\n\n".join([doc.page_content for doc in docs])
                sources = [clean_context(doc.page_content) for doc in docs]
                
                # Generate answer
                with st.spinner("Generating answer..."):
                    answer = query_model(context, user_question)
                    
                    # Update history
                    st.session_state.history[-1].update({
                        "answer": answer,
                        "sources": sources
                    })
                    
                    # Display answer
                    st.write(answer)
        
        # Show sources
        if st.session_state.history[-1].get("sources"):
            with st.expander("📚 View Source Passages"):
                for i, source in enumerate(st.session_state.history[-1]["sources"], 1):
                    st.markdown(f"**Passage {i}:**")
                    st.write(source)
                    if i < len(st.session_state.history[-1]["sources"]):
                        st.divider()

# Clear chat button
if st.session_state.history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.history = []
        st.rerun()