from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
# Load API Key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=api_key)
def create_faiss_from_pdf(pdf_path, save_path="saved_faiss"):
    try:
        # 1. Load the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 2. Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        # 3. Create FAISS vector store
        vectorstore = FAISS.from_documents(docs, embedding_model)

        # 4. Save FAISS store
        vectorstore.save_local(save_path)
        print(f"FAISS vector store created and saved at '{save_path}'")
    except Exception as e:
        print(f" Error: {str(e)}")

# Use your actual PDF file name
create_faiss_from_pdf("PF Lec no 5.pdf")