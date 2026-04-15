#REQ_F004: Retrieve relevant documents chunks and generate cited answers

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma

#same embedding model as ETL so query vectors match stored documents vectors
from langchain_huggingface import HuggingFaceEmbeddings 


from langchain_community.llms import Ollama
 # connects to the local Ollama server to run the Mistral model
  # ── AZURE SWAP ──
  # Replace Ollama with:
  #   from langchain_openai import AzureChatOpenAI
  # Requires: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
  # ── END AZURE SWAP ──

#load .env file values
load_dotenv()

#function to load ChromaDB knowledge base data
def load_vector_store() -> Chroma:

    embedding_model = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL") #same embedding model as ETL

        #--Azure swap--
        #HuggingFaceEmbeddings function later replace with:
        #AzureOpenAIEmbeddings(
        #       azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        #       azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #       api_key=os.getenv("AZURE_OPENAI_API_KEY")
        #   )
        # Requires: AZURE_* values in .env
        # ── END AZURE SWAP ──
    )

    vector_store = Chroma(
        persist_directory=os.getenv("CHROMA_DB_PATH"), #folder where ChromaDB saved
        embedding_function=embedding_model, #embed incoming queries
        collection_name=os.getenv("CHROMA_COLLECTION_NAME")
    )

    return vector_store


#function to find most relevant document chunks for a user question before answer generation
def retrieve_relevant_chunks(question:str, top_k:int = 5) -> list:
    vector_store = load_vector_store()
    results = vector_store.similarity_search(question,k=top_k)

    return results

#function to convert retrieved LangChain Document chunks into plain text context for LLM and extract source filenames for citation.
def build_context_and_sources(chunks:list) ->tuple[str,list]:
    #store text content for each retrieved chunk
    context_parts = []

    #store source filenames for citation display
    sources = []

    for chunk in chunks:
        context_parts.append(chunk.page_content)

        source_name = chunk.metadata.get("source","Unknown source")
        sources.append(source_name)

    #merge chunks into one prompt context block
    context_text = "\n\n".join(context_parts)

    #remove duplicates while preserving order
    unique_sources = list(dict.fromkeys(sources))

    #return in tuple
    return context_text,unique_sources

#function to retrieving relevant evidence, building a grounded prompt, calling LLM, and returning both answer and sources
def generate_answer(question:str) -> dict:
    #retrieve most relevant document chunks
    chunks = retrieve_relevant_chunks(question)

    #prepare context and citation
    context_text, sources = build_context_and_sources(chunks)

    #system prompt (grounded rules for anti hallucination)
    prompt = f"""
    You are an internal knowledge assistant for Centific Malaysia.
    Answer the user's question using only the context provided below.
    If the answer is not clearly supported by the context, say that the information was not found in the available documents.
    Keep the answer clear and professional.

    Context:
    {context_text}

    Question:
    {question}
    """.strip()

    llm = Ollama(
        base_url= os.getenv("OLLAMA_BASE_URL"),
        model = os.getenv("OLLAMA_MODEL")
    )
    # ── AZURE SWAP ──
    # Replace Ollama(...) above with:
    #   AzureChatOpenAI(
    #       azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    #       azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #       api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #       api_version="2024-02-01"
    #   )
    # Requires: AZURE_* values in .env
    # ── END AZURE SWAP ──

    #send grounded prompt to LLM and get response
    answer = llm.invoke(prompt)

    return {
        "question": question,
        "answer":answer,
        "sources":sources
    }