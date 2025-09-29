import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import markdown

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Request model for API
class AssistantRequest(BaseModel):
    query: str
    name: str

# Embeddings and model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=api_key)
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite",api_key=api_key)

#prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "query", "name"],
    template="""
    You are a helpful **PDF Assistant**, designed to read, analyze, and answer questions 
    based on the content of uploaded PDF documents.

    ---
    ### **Your Guidelines:**  
    1. Use the provided PDF context to answer the question as clearly as possible.  
    2. Provide **well-structured, detailed, and easy-to-understand responses**.  
    3. If the context does not include enough information, respond politely with:  
       _"I don’t have enough information from this PDF to answer that. Please check the document for more details."_  
    4. Avoid making up information not present in the PDF.  
    5. Use a **friendly, natural, and professional tone** (not too robotic).  
    6. Break down complex answers into **bullet points or sections** for clarity.  
    7. Add **examples, explanations, or summaries** when useful.  
    8. Address the user by their name to make the response more engaging.  

    ---
    ### **Context from PDF:**  
    {context}  

    ### **User Query:**  
    {query}  

    ### **Username:**  
    {name}

    ---
    ### **Your Response:**  

    Start naturally, for example:  
    - "You're asking about something in the PDF? Here's the breakdown 👇"  
    - "Great question, {name}! Let’s look at what the PDF says..."  

    Then provide your detailed answer.
    """
)

# Load FAISS vector store
def load_vectore_store(path) -> FAISS:
     try:
         vectore_store= FAISS.load_local(path,embeddings=embedding_model,allow_dangerous_deserialization=True)

         return {"success":True,"message":vectore_store,"status":200}
     except Exception as e:
         return {"success":False,"message":f"Some Error Encountoured in Loading Vectore store {str(e)}","status":400}
     _google_genai

# Find similar text from vector store
def find_similar_text(user_query):
    try:
        response = load_vectore_store("saved_faiss")
        vectore_store = response['message']
        retriever  = vectore_store.as_retriever()
        text = retriever.invoke(user_query)
        return {"success":True,"message":text,"status":200}
    except Exception as e:
         return {"success":False,"message":f"Some Error Encountoured in retrieving similar text {str(e)}","status":400}

# Find similar text from vector store
def get_response(user_query,user_name):
    similar_text = find_similar_text(user_query)
    prompt = prompt_template.format(query = user_query,context=similar_text,name=user_name)
    response = model.invoke(prompt)
    return markdown.markdown(response.content)

# print(get_response("Hello","Maida"))

# Interactive CLI Chat Loop
while True:
    user_query = input("enter your query:\n")
    if user_query.lower()=="exit":
        print("Thank you for using the chatbot.")
        break
    print(get_response(user_query=user_query,user_name="User"))

# FastAPI Router Endpoint
router= APIRouter()    
  
@router.post("/assistant")
def get_assistant_response(request: AssistantRequest):
    try:
        response = get_response(request.query, request.name)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))