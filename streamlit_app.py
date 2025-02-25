from fastapi import FastAPI
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

app = FastAPI()

url = 'https://info.eminenceorganics.com/skin-care-routine'

@st.cache_resource
def load_webpage_content(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Create vector store
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def create_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="llama2")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

st.title("RAG Q&A System")

# URL input
urls = st.text_area("Enter URLs (one per line)", height=100)
url_list = [url.strip() for url in urls.split("\n") if url.strip()]

if st.button("Load URLs"):
    with st.spinner("Loading and processing URLs..."):
        vectorstore = load_vectorstore(url_list)
        st.session_state.qa_chain = create_rag_pipeline(vectorstore)
    st.success("URLs loaded and processed!")

# Question input
question = st.text_input("Ask a question")

if question and 'qa_chain' in st.session_state:
    with st.spinner("Generating answer..."):
        response = st.session_state.qa_chain.invoke({"query": question})
        st.write("Answer:", response['result'])
