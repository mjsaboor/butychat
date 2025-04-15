# app.py - FastAPI Backend
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Models
class ProfileAnswers(BaseModel):
    skin_type: str
    makeup_frequency: str
    skin_concerns: List[str]
    
class QueryRequest(BaseModel):
    query: str
    profile: ProfileAnswers

class QueryResponse(BaseModel):
    result: str

# Initialize FastAPI
app = FastAPI(title="Beauty RAG System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG pipeline
qa_chain = None
vectorstore = None

# Functions
def load_pdf_content(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=512, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(chunks):
    model_name = 'HooshvareLab/bert-fa-base-uncased'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def create_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="gemma3:1b")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def format_query_with_profile(user_query, profile):
    profile_context = f"\nنوع پوست: {profile.skin_type}\n" \
                      f"تعداد دفعات آرایش در هفته: {profile.makeup_frequency}\n" \
                      f"نگرانی‌های پوستی: {', '.join(profile.skin_concerns)}"
    full_query = f"پاسخ را فقط به فارسی بده. پاسخ را کامل و قدم به قدم توضیح بده.\n{profile_context}\nسوال کاربر: {user_query}"
    return full_query

@app.on_event("startup")
async def startup_event():
    global qa_chain, vectorstore
    try:
        file_path = os.getenv('PDF_PATH', 'farsi-skin routine.pdf')
        documents = load_pdf_content(file_path)
        chunks = split_documents(documents)
        vectorstore = create_vectorstore(chunks)
        qa_chain = create_rag_pipeline(vectorstore)
    except Exception as e:
        print(f"Error during initialization: {e}")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not qa_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        formatted_query = format_query_with_profile(request.query, request.profile)
        response = qa_chain.invoke({"query": formatted_query})
        return {"result": response['result']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    if qa_chain:
        return {"status": "healthy"}
    else:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": "RAG system not initialized"})

# Mount static files for frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
