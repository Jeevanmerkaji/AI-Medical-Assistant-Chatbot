from flask import Flask, render_template, jsonify, request
from src.helper import loader_pdf_file, filter_to_minimal_docs, split_text, download_hugging_face_embeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from src.prompt import *
import os 



app = Flask(__name__)

GROQ_API_KEY =  os.environ.get('GROQ_API_KEY')
os.environ["GROQ_API_KEY"] =  GROQ_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "AI assisted chatbot for the Medical Diagnosis"



extracted_data =  loader_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks =  split_text(filter_data)

texts = [doc.page_content for doc in text_chunks]
metadatas = [doc.metadata for doc in text_chunks]  # Optional

chat_model = ChatGroq(
    model_name="llama3-70b-8192"
)

prompt =  ChatPromptTemplate.from_messages(
    [
        ("system" , system_prompt),
        ("human","{input}")
    ]
)

# Create FAISS vector store
vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
retriever  = vectorstore.as_retriever(search_type = "similarity", seacr_kwargs ={"k":3})
question_answer_chain =  create_stuff_documents_chain(chat_model, system_prompt)
rag_chain =  create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get" , methods =["GET", "POST"])
def chat():
    msg =  request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response['answer'])

    return str(response["answer"])



if __name__ == "__main__":
    app.run(host="0.0.0.0", port =8080, debug= True)