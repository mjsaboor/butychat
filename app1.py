from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dataclasses import dataclass
from typing import List, Optional
import os
import threading
import json
import time
from threading import Thread
from queue import Queue

# Updated imports to fix deprecation warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_qdrant import Qdrant  # Updated import
from langchain_ollama import OllamaLLM  # Updated import
from langchain.chains import RetrievalQA
import qdrant_client

# Configuration settings
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'beauty_knowledge')
PDF_PATH = os.getenv('PDF_PATH', 'farsi-skin routine.pdf')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
LLM_MODEL = os.getenv('LLM_MODEL', 'gemma3:1b')
LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', 180))  # Increased timeout for LLM responses

# Data models using dataclasses
@dataclass
class ProfileAnswers:
    skin_type: str
    makeup_frequency: str
    skin_concerns: List[str]

# Global variables for RAG pipeline
qa_chain = None
vectorstore = None
initialization_complete = False
# Dictionary to store background tasks
processing_tasks = {}

# Flask app initialization
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Functions for RAG pipeline
def load_pdf_content(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
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
        prefer_grpc=False  # Use REST API instead of gRPC
    )
    
    # Create vectorstore using Qdrant
    vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION_NAME,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        prefer_grpc=False  # Use REST API instead of gRPC
    )
    
    return vectorstore

def create_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # Initialize LLM with higher timeout
    llm = OllamaLLM(model=LLM_MODEL, timeout=LLM_TIMEOUT)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def format_query_with_profile(user_query, profile):
    profile_context = f"\nنوع پوست: {profile['skin_type']}\n" \
                      f"تعداد دفعات آرایش در هفته: {profile['makeup_frequency']}\n" \
                      f"نگرانی‌های پوستی: {', '.join(profile['skin_concerns'])}"
    full_query = f"پاسخ را فقط به فارسی بده. پاسخ را کامل و قدم به قدم توضیح بده.\n{profile_context}\nسوال کاربر: {user_query}"
    return full_query

def initialize_vectorstore():
    """Initializes the vectorstore separately"""
    try:
        documents = load_pdf_content(PDF_PATH)
        chunks = split_documents(documents)
        create_vectorstore(chunks)
        print(f"Vectorstore initialization complete. Collection: {QDRANT_COLLECTION_NAME}")
    except Exception as e:
        print(f"Error during vectorstore initialization: {e}")
        raise

def get_rag_pipeline():
    """Gets or initializes the RAG pipeline"""
    global qa_chain, vectorstore, initialization_complete
    
    if not initialization_complete:
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
            initialization_complete = True
        except Exception as e:
            print(f"Error getting RAG pipeline: {e}")
    
    return qa_chain

def process_query_task(query_id, formatted_query, result_queue):
    """Background task to process a query"""
    try:
        chain = get_rag_pipeline()
        if not chain:
            result_queue.put((query_id, {"error": "RAG system not initialized"}, 503))
            return
            
        print(f"Processing query {query_id}...")
        start_time = time.time()
        response = chain.invoke({"query": formatted_query})
        elapsed = time.time() - start_time
        print(f"Query {query_id} completed in {elapsed:.2f} seconds")
        result_queue.put((query_id, {"result": response['result']}, 200))
    except Exception as e:
        print(f"Error processing query {query_id}: {str(e)}")
        result_queue.put((query_id, {"error": f"Error processing your query: {str(e)}"}, 500))

# Check startup
def startup_check():
    try:
        client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections()
        if QDRANT_COLLECTION_NAME not in [c.name for c in collections.collections]:
            print(f"Warning: Collection '{QDRANT_COLLECTION_NAME}' not found. Run initialization first.")
    except Exception as e:
        print(f"Error connecting to Qdrant during startup: {e}")

# Routes
@app.route('/api/query', methods=['POST'])
def process_query():
    # Run startup check on first request
    if not getattr(app, '_got_first_request', False):
        app._got_first_request = True
        startup_check()
    
    try:
        # Parse JSON request
        request_data = request.get_json()
        user_query = request_data.get('query')
        profile = request_data.get('profile')
        
        # Validate request data
        if not user_query or not profile:
            return jsonify({"error": "Missing query or profile data"}), 400
        
        formatted_query = format_query_with_profile(user_query, profile)
        
        # Generate a unique ID for this query
        query_id = str(int(time.time() * 1000))
        
        # Create a queue for the result
        result_queue = Queue()
        
        # Start a background thread to process the query
        thread = Thread(target=process_query_task, args=(query_id, formatted_query, result_queue))
        thread.start()
        
        # Store the thread and queue
        processing_tasks[query_id] = {
            "thread": thread,
            "queue": result_queue,
            "start_time": time.time()
        }
        
        # Return the query ID immediately
        return jsonify({
            "query_id": query_id, 
            "status": "processing",
            "message": "Your query is being processed. Check status at /api/query/status/" + query_id
        })
    except Exception as e:
        print(f"Query setup error: {str(e)}")
        return jsonify({"error": "Error setting up your query"}), 500

@app.route('/api/query/status/<query_id>', methods=['GET'])
def check_query_status(query_id):
    """Check the status of a processing query"""
    if query_id not in processing_tasks:
        return jsonify({"error": "Query not found"}), 404
    
    task = processing_tasks[query_id]
    queue = task["queue"]
    
    # Check if result is available
    if not queue.empty():
        # Get result and remove task
        _, result, status_code = queue.get()
        processing_tasks.pop(query_id, None)
        return jsonify(result), status_code
    
    # If thread is still running, return status
    if task["thread"].is_alive():
        elapsed = time.time() - task["start_time"]
        return jsonify({
            "status": "processing",
            "elapsed_seconds": elapsed,
            "message": "Your query is still being processed"
        })
    
    # If thread is done but queue is empty (shouldn't happen)
    return jsonify({"error": "Processing completed but no result found"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check Qdrant connection
        client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections()
        collection_exists = QDRANT_COLLECTION_NAME in [c.name for c in collections.collections]
        
        health_data = {
            "status": "healthy" if qa_chain and collection_exists else "unhealthy",
            "components": {
                "qdrant": "available" if collection_exists else "collection not found",
                "qa_chain": "available" if qa_chain else "unavailable"
            }
        }
        
        status_code = 200 if health_data["status"] == "healthy" else 503
        return jsonify(health_data), status_code
    except Exception as e:
        return jsonify({
            "status": "unhealthy", 
            "reason": f"Qdrant connection error: {str(e)}"
        }), 503

@app.route('/admin/initialize', methods=['POST'])
def admin_initialize():
    try:
        # Run initialization in a background thread
        thread = threading.Thread(target=initialize_vectorstore)
        thread.start()
        return jsonify({"status": "success", "message": "Vectorstore initialization started"})
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Initialization failed: {str(e)}"
        }), 500

# Serve static files
@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    # Run startup check at application start
    print("Running initial startup check...")
    startup_check()
    
    # Check if we should only initialize
    if os.environ.get("INITIALIZE_ONLY", "0") == "1":
        initialize_vectorstore()
    else:
        app.run(host="0.0.0.0", port=8000, debug=True)
