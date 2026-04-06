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

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# loads local sentence-transformer model for embeddings

# ── AZURE SWAP ──
# Replace HuggingFaceEmbeddings with:
#   from langchain_openai import AzureOpenAIEmbeddings
# Requires: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT in .env
# ── END AZURE SWAP ──


#load all values from .env into environment variables
load_dotenv() 