import streamlit as st
import os
import threading
import json
import time
from typing import List, Optional
from dataclasses import dataclass
import asyncio
from queue import Queue
from threading import Thread

# Updated imports to fix deprecation warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import qdrant_client

# Configuration settings
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'beauty_knowledge')
PDF_PATH = os.getenv('PDF_PATH', 'farsi-skin routine.pdf')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
LLM_MODEL = os.getenv('LLM_MODEL', 'gemma3:1b')
LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', 180))

# Data models using dataclasses
@dataclass
class ProfileAnswers:
    skin_type: str
    makeup_frequency: str
    skin_concerns: List[str]

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'initialization_complete' not in st.session_state:
    st.session_state.initialization_complete = False
if 'processing_queries' not in st.session_state:
    st.session_state.processing_queries = {}

# Functions for RAG pipeline
def load_pdf_content(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        raise

def split_documents(documents, chunk_size=512, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Initialize Qdrant client
    client = qdrant_client.QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=False
    )
    
    # Create vectorstore using Qdrant
    vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION_NAME,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        prefer_grpc=False
    )
    
    return vectorstore

def create_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model=LLM_MODEL, timeout=LLM_TIMEOUT)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def format_query_with_profile(user_query, profile):
    profile_context = f"\nنوع پوست: {profile['skin_type']}\n" \
                      f"تعداد دفعات آرایش در هفته: {profile['makeup_frequency']}\n" \
                      f"نگرانی‌های پوستی: {', '.join(profile['skin_concerns'])}"
    full_query = f"پاسخ را فقط به فارسی بده. پاسخ را کامل و قدم به قدم توضیح بده.\n{profile_context}\nسوال کاربر: {user_query}"
    return full_query

@st.cache_resource
def initialize_vectorstore():
    """Initializes the vectorstore - cached to avoid re-initialization"""
    try:
        with st.spinner("Initializing vectorstore..."):
            documents = load_pdf_content(PDF_PATH)
            chunks = split_documents(documents)
            create_vectorstore(chunks)
            st.success(f"Vectorstore initialization complete. Collection: {QDRANT_COLLECTION_NAME}")
            return True
    except Exception as e:
        st.error(f"Error during vectorstore initialization: {e}")
        return False

@st.cache_resource
def get_rag_pipeline():
    """Gets or initializes the RAG pipeline - cached for performance"""
    try:
        # Create embedding function
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Connect to existing Qdrant collection
        vectorstore = Qdrant(
            client=qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=False),
            collection_name=QDRANT_COLLECTION_NAME,
            embeddings=embeddings
        )
        
        # Create QA chain
        qa_chain = create_rag_pipeline(vectorstore)
        return qa_chain
    except Exception as e:
        st.error(f"Error getting RAG pipeline: {e}")
        return None

def check_system_health():
    """Check system health"""
    try:
        client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections()
        collection_exists = QDRANT_COLLECTION_NAME in [c.name for c in collections.collections]
        return collection_exists, None
    except Exception as e:
        return False, str(e)

def process_query_sync(formatted_query):
    """Process query synchronously"""
    try:
        chain = get_rag_pipeline()
        if not chain:
            return None, "RAG system not initialized"
        
        response = chain.invoke({"query": formatted_query})
        return response['result'], None
    except Exception as e:
        return None, f"Error processing your query: {str(e)}"

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Beauty Consultation RAG System",
        page_icon="💄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💄 Beauty Consultation RAG System")
    st.markdown("---")
    
    # Sidebar for system status and controls
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Health check
        collection_exists, error = check_system_health()
        if collection_exists:
            st.success("✅ Qdrant Connected")
            st.success(f"✅ Collection '{QDRANT_COLLECTION_NAME}' Available")
        else:
            st.error("❌ Qdrant Connection Failed")
            if error:
                st.error(f"Error: {error}")
        
        st.markdown("---")
        
        # Initialize vectorstore button
        st.header("🚀 System Setup")
        if st.button("Initialize Vectorstore", type="primary"):
            success = initialize_vectorstore()
            if success:
                # Clear cache to reload the pipeline
                st.cache_resource.clear()
        
        st.markdown("---")
        
        # Configuration display
        st.header("⚙️ Configuration")
        st.text(f"Qdrant Host: {QDRANT_HOST}:{QDRANT_PORT}")
        st.text(f"Collection: {QDRANT_COLLECTION_NAME}")
        st.text(f"PDF Path: {PDF_PATH}")
        st.text(f"LLM Model: {LLM_MODEL}")
        st.text(f"Embedding Model: {EMBEDDING_MODEL.split('/')[-1]}")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("👤 User Profile")
        
        # Skin type selection
        skin_type = st.selectbox(
            "نوع پوست (Skin Type)",
            ["خشک (Dry)", "چرب (Oily)", "ترکیبی (Combination)", "حساس (Sensitive)", "عادی (Normal)"]
        )
        
        # Makeup frequency
        makeup_frequency = st.selectbox(
            "تعداد دفعات آرایش در هفته (Makeup Frequency per Week)",
            ["هیچوقت (Never)", "کمتر از ۳ بار (Less than 3 times)", 
             "۳-۵ بار (3-5 times)", "روزانه (Daily)"]
        )
        
        # Skin concerns (multi-select)
        skin_concerns = st.multiselect(
            "نگرانی‌های پوستی (Skin Concerns)",
            ["آکنه (Acne)", "لک (Dark Spots)", "چروک (Wrinkles)", 
             "خشکی (Dryness)", "چربی زیاد (Excess Oil)", "حساسیت (Sensitivity)",
             "منافذ باز (Large Pores)", "پوست خسته (Tired Skin)"]
        )
        
        # Display current profile
        if skin_type and makeup_frequency:
            st.subheader("📋 Current Profile")
            profile_data = {
                "skin_type": skin_type,
                "makeup_frequency": makeup_frequency,
                "skin_concerns": skin_concerns
            }
            st.json(profile_data)
    
    with col2:
        st.header("💬 Beauty Consultation")
        
        # Query input
        user_query = st.text_area(
            "سوال خود را بپرسید (Ask your question in Persian)",
            height=100,
            placeholder="مثال: چه محصولاتی برای پوست خشک مناسب است؟"
        )
        
        # Process button
        if st.button("پردازش سوال (Process Query)", type="primary"):
            if not user_query.strip():
                st.warning("لطفاً سوال خود را وارد کنید (Please enter your question)")
            elif not skin_type or not makeup_frequency:
                st.warning("لطفاً پروفایل خود را کامل کنید (Please complete your profile)")
            else:
                # Create profile dictionary
                profile = {
                    "skin_type": skin_type,
                    "makeup_frequency": makeup_frequency,
                    "skin_concerns": skin_concerns
                }
                
                # Format query with profile
                formatted_query = format_query_with_profile(user_query, profile)
                
                # Process query
                with st.spinner("در حال پردازش سوال شما... (Processing your query...)"):
                    result, error = process_query_sync(formatted_query)
                
                if error:
                    st.error(f"خطا: {error}")
                else:
                    st.success("✅ پاسخ آماده است!")
                    st.markdown("### 📝 پاسخ (Answer)")
                    st.markdown(result)
        
        # Query history in session state
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        # Display recent queries
        if st.session_state.query_history:
            st.markdown("---")
            st.subheader("📚 Recent Queries")
            for i, (query, answer) in enumerate(reversed(st.session_state.query_history[-3:])):
                with st.expander(f"Query {len(st.session_state.query_history)-i}: {query[:50]}..."):
                    st.markdown(f"**Question:** {query}")
                    st.markdown(f"**Answer:** {answer}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Beauty Consultation RAG System | Powered by LangChain & Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
