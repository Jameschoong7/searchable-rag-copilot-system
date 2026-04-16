# REQ_F001: ETL pipeline — Extract, Transform, Load for local documents
# REQ_F002: Chunking and embedding for vector search

import os
from pathlib import Path
from dotenv import load_dotenv  #read .env config file

from langchain_community.document_loaders import TextLoader, PyPDFLoader
#TextLoader read .txt files into LangChain Document objects
#PyPDFLoader read .pdf files into LangChain Document objects

from langchain_text_splitters import RecursiveCharacterTextSplitter
#split long documents into smaller overlapping chunks

from langchain_community.vectorstores import Chroma
#ChromaDB wrapper - handles storing and searching embeddings

from langchain_huggingface import HuggingFaceEmbeddings
# loads local sentence-transformer model for embeddings

# ── AZURE SWAP ──
# Replace HuggingFaceEmbeddings with:
#   from langchain_openai import AzureOpenAIEmbeddings
# Requires: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT in .env
# ── END AZURE SWAP ──


#load all values from .env into environment variables
load_dotenv() 

#function to handle E in ETL
def load_documents(folder_path:str) -> list:
    #(REQ_F001) E in ETL, read all .txt and .pdf files from given folder
    #Return a list of LangChain Document object, each holding file text + metadata
    
    #list for storing all loaded Document objects
    documents = []

    #converet string path to Path object 
    folder = Path(folder_path)

    #loop through every file in the folder
    for file_path in folder.iterdir():
        #validate file extension if it's .txt
        if file_path.suffix == ".txt":
            #create text loader
            loader = TextLoader(str(file_path), encoding="utf-8")

            #load and add to list
            documents.extend(loader.load())
        
        #if it's .pdf
        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
    
    #feedback for the progress
    print(f"Loaded {len(documents)} document(s) from {folder_path}")
    return documents

#function to handle T in ETL
def chunk_documents(documents: list) -> list:
    #(REQ_F002): splits large documents into smaller overlapping chunks.
    #smaller chunks means more precise retrieval. 
    #overlap means no lost context at boundaries
    
    splitter = RecursiveCharacterTextSplitter(
        #max characters per chunk
        chunk_size=900,

        #overlap between consecutive chunks to preserve context
        chunk_overlap=100,

        #tries to split at paragraph first, then lines, then words
        separators=["\n\n","\n"," ",""]

    )

    #new list of smaller Document objects
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks

#function to handle second half of T and the L in ETL
def embed_and_store(chunks: list, db_path: str, collection_name: str) -> Chroma:
    #Transform + Load step (REQ_F002): embeds each chunk into a vector
    #persists all vectors to ChromaDB for later retrieval

    embedding_model = HuggingFaceEmbeddings(
        #read model from .env
        model_name = os.getenv("EMBEDDING_MODEL"),

        # ── AZURE SWAP ──
        # Replace HuggingFaceEmbeddings(...) with:
        #   AzureOpenAIEmbeddings(
        #       azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        #       azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #       api_key=os.getenv("AZURE_OPENAI_API_KEY")
        #   )
        # ── END AZURE SWAP ──
    )

    #embed all chunks and write to ChromaDB in one step
    vector_store = Chroma.from_documents(
        #list of chunked Document objects
        documents=chunks,

        #model that converts text to vectors
        embedding=embedding_model,

        #logical name for this set of documents
        collection_name=collection_name,
        
        #folder where ChromaDB save its file
        persist_directory=db_path

    )

    print(f"Stored {len(chunks)} chunks to ChromaDB at {db_path}")
    return vector_store


#main entry point
if __name__ =="__main__":
    #orchestrates full ETL pipeline when run directly
    #reads config from .env, processes all documents, stores to ChromaDB

    #read config from .env
    docs_path = os.getenv("DOCUMENTS_PATH")
    db_path = os.getenv("CHROMA_DB_PATH")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME")

    print("Starting ETL pipeline...")

    #step 1: Extract (E)
    documents = load_documents(docs_path)

    #step 2: Transform (T)
    chunks = chunk_documents(documents)

    #step 3: Load
    embed_and_store(chunks, db_path, collection_name)

    print("ETL pipeline complete. ChromaDB is ready")