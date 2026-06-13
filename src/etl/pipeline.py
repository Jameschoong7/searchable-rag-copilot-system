# REQ_F001: ETL pipeline — Extract, Transform, Load for local documents
# REQ_F002: Chunking and embedding for vector search

import shutil
import os
from pathlib import Path
from dotenv import load_dotenv  #read .env config file

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from docx import Document as DocxDocument
#TextLoader read .txt files into LangChain Document objects
#PyPDFLoader read .pdf files into LangChain Document objects
#Docx allow use of python-docx to extract text then wrap into LangChain Document objects

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


def extract_docx_text(file_path: Path) -> str:
    """Extract searchable paragraph and table text from a DOCX file."""
    docx_file = DocxDocument(str(file_path))
    text_parts = []

    for paragraph in docx_file.paragraphs:
        paragraph_text = paragraph.text.strip()

        if paragraph_text:
            text_parts.append(paragraph_text)

    for table in docx_file.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            )

            if row_text:
                text_parts.append(row_text)

    return "\n".join(text_parts)

#function to handle E in ETL
def load_documents(folder_path:str) -> tuple[list, int]:
    #(REQ_F001) E in ETL, read all .txt and .pdf files from given folder
    #Return a list of LangChain Document object, each holding file text + metadata
    
    #list for storing all loaded Document objects
    documents = []

    #count physical files separately because PDFs can produce one Document per page
    source_file_count = 0

    #converet string path to Path object 
    folder = Path(folder_path)

    #loop through every file in the folder
    for file_path in folder.iterdir():
        if not file_path.is_file():
          continue

        suffix = file_path.suffix.lower()

        if suffix not in [".txt", ".pdf", ".docx"]:
            continue

        source_file_count += 1

        if suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

        elif suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

        elif suffix == ".docx":
            docx_text = extract_docx_text(file_path)

            if docx_text.strip():
                documents.append(
                    Document(
                        page_content=docx_text,
                        metadata={"source": str(file_path)},
                    )
                )
    
    #feedback for the progress
    print(
        f"Loaded {source_file_count} file(s) and "
        f"{len(documents)} document object(s) from {folder_path}"
    )
    return documents, source_file_count

#function to handle T in ETL
def chunk_documents(documents: list) -> list:
    #(REQ_F002): splits large documents into smaller overlapping chunks.
    #smaller chunks means more precise retrieval. 
    #overlap means no lost context at boundaries
    
    splitter = RecursiveCharacterTextSplitter(
        #max characters per chunk
        chunk_size=900,

        #overlap between consecutive chunks to preserve context
        chunk_overlap=250,

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


def rebuild_vector_store(
    docs_path: str | None = None,
    db_path: str | None = None,
    collection_name: str | None = None,
) -> dict:
    """Rebuild the local ChromaDB vector store from the simulated documents folder."""
    docs_path = docs_path or os.getenv("DOCUMENTS_PATH")
    db_path = db_path or os.getenv("CHROMA_DB_PATH")
    collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME")

    if docs_path is None or db_path is None or collection_name is None:
        raise ValueError("DOCUMENTS_PATH, CHROMA_DB_PATH, and CHROMA_COLLECTION_NAME are required.")

    chroma_directory = Path(db_path)

    if chroma_directory.exists():
        shutil.rmtree(chroma_directory)

    documents, source_file_count = load_documents(docs_path)
    chunks = chunk_documents(documents)
    embed_and_store(chunks, db_path, collection_name)

    return {
        "documents_indexed": source_file_count,
        "document_objects_loaded": len(documents),
        "chunks_indexed": len(chunks),
        "collection_name": collection_name,
        "db_path": db_path,
    }


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
    documents, source_file_count = load_documents(docs_path)

    #step 2: Transform (T)
    chunks = chunk_documents(documents)

    #step 3: Load
    embed_and_store(chunks, db_path, collection_name)

    print("ETL pipeline complete. ChromaDB is ready")