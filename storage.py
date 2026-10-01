from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


docs=PyMuPDFLoader("loader\\langchain.pdf") 
data=docs.load()

split=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks=split.split_documents(data)

emb_model=HuggingFaceEmbeddings(
     model="sentence-transformers/all-mpnet-base-v2"
)

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=emb_model,
    persist_directory="chroma_db"
)


