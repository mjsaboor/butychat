from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import requests
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

def load_pdf_content(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

    # Step 2: Split documents into smaller chunks
def split_documents(documents, chunk_size=512, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    length_function=len)
   #text_splitter = TokenTextSplitter(
   #    chunk_size=chunk_size,  # Tokens, not characters
   #    chunk_overlap=chunk_overlap,
   #    encoding_name="cl100k_base")
    return text_splitter.split_documents(documents)

# Step 3: Create a vector store for retrieval
def create_vectorstore(chunks):
    #model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model_name = 'HooshvareLab/bert-fa-base-uncased'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)  # Ensure you have sentence-transformers installed
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Step 4: Create the RAG pipeline
def create_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="deepseek-r1:32b-qwen-distill-q4_K_M")  # Ensure Ollama is running
 
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def main():
    # Load webpage content
    file_path = '/workspaces/butychat/farsi-skin routine.pdf'  # Replace with your PDF file path
    print(f"Loading content from {file_path}...")
    documents = load_pdf_content(file_path)

    # Split content into chunks
    print("Splitting documents into chunks...")
    chunks = split_documents(documents)

    # Create a vector store
    print("Creating vector store for retrieval...")
    vectorstore = create_vectorstore(chunks)

    # Create the RAG pipeline
    print("Setting up the RAG pipeline...")
    qa_chain = create_rag_pipeline(vectorstore)

    if qa_chain is None:
        return

    # Define the Persian prompt template
    persian_prompt = PromptTemplate(
        input_variables=["query"],  # Placeholder for user query
        template="پاسخ را فقط به فارسی بده. پاسخ را کامل و قدم به قدم توضیح بده. سوال کاربر: {query}"
    )

    print("Chatbot is ready! Ask your questions:")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Format the query with the Persian instruction
        query = persian_prompt.format(query=user_query)

        # Get the response from the model
        response = qa_chain.invoke({"query": query})

        print(f"Bot: {response['result']}")

if __name__ == "__main__":
    main()
