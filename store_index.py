from dotenv import load_dotenv
import os 
from src.helper import loader_pdf_file, filter_to_minimal_docs, split_text, download_hugging_face_embeddings
from langchain.vectorstores import FAISS
load_dotenv()


GROQ_API_KEY =  os.environ.get('GROQ_API_KEY')

os.environ["GROQ_API_KEY"] =  GROQ_API_KEY


extracted_data =  loader_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks =  split_text(filter_data)

embeddings =  download_hugging_face_embeddings()


# Let's assume text_chunks is a list of Document objects
texts = [doc.page_content for doc in text_chunks]
metadatas = [doc.metadata for doc in text_chunks]  # Optional

# Create FAISS vector store
vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)