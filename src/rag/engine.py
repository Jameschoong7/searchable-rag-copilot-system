#REQ_F004: Retrieve relevant documents chunks and generate cited answers

import re
import json
from pathlib import Path
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "data/simulated/document_metadata.json"

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


def load_document_metadata() -> list[dict]:
    """Load simulated document metadata used for ACL filtering before retrieval."""
    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def can_access_document(document: dict, role: str, department: str) -> bool:
    """Check whether a role and department can retrieve a document."""
    if role == "System Admin":
        return True

    return (
        role in document["allowed_roles"]
        and department in document["allowed_departments"]
    )


def document_matches_filters(
    document: dict,
    department_filter: str | None,
    file_type_filter: str | None,
) -> bool:
    """Check whether a document matches optional user-selected search filters."""
    if department_filter and department_filter != "All":
        if document["department"] != department_filter:
            return False

    if file_type_filter and file_type_filter != "All":
        if document["file_type"] != file_type_filter:
            return False

    return True


def query_matches_document(question: str, document: dict) -> bool:
    """Check whether a user question appears to target a document's metadata."""
    ignored_words = {
        "what",
        "when",
        "where",
        "which",
        "with",
        "from",
        "that",
        "this",
        "does",
        "have",
        "policy",
        "policies",
        "requirement",
        "requirements",
        "procedure",
        "process",
}
    
    query_keywords = {
        word
        for word in re.findall(r"[a-z0-9]+", question.lower())
        if len(word) >= 4 and word not in ignored_words
    }

    searchable_metadata = " ".join(
        [
            document["title"],
            document["category"],
            " ".join(document["tags"]),
        ]
    ).lower()

    metadata_words = set(re.findall(r"[a-z0-9]+", searchable_metadata))

    return bool(query_keywords & metadata_words)


def get_allowed_source_path(
    role: str,
    department: str,
    department_filter: str | None = None,
    file_type_filter: str | None = None,
) -> list[str]:
    """Return Chroma source paths that the current user is allowed to retrieve."""
    documents = load_document_metadata()

    allowed_filenames = []

    for document in documents:
        if not can_access_document(document, role, department):
            continue

        if not document_matches_filters(
            document,
            department_filter,
            file_type_filter,
        ):
            continue

        allowed_filenames.append(document["filename"])

    return [
        str(PROJECT_ROOT / "data/simulated" / filename)
        for filename in allowed_filenames
    ]


def retrieve_relevant_chunks(
    question:str,
    role: str,
    department: str,
    department_filter: str | None = None,
    file_type_filter: str | None = None,
    top_k:int = 5,
) -> list:
    """Retrieve only chunks from documents allowed for the user's role and department."""
    vector_store = load_vector_store()
    allowed_sources = get_allowed_source_path(
        role,
        department,
        department_filter,
        file_type_filter
    )

    if not allowed_sources:
        return []
    
    scored_results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=top_k,
        filter={"source": {"$in": allowed_sources}},
    )

    minimum_relevance_score = 0.30

    return [
        document
        for document, score in scored_results
        if score >= minimum_relevance_score
    ]


def find_restricted_matching_documents(
    question: str,
    role: str,
    department: str,
) -> list[dict]:
    """Find documents that match the question but are not accessible to the user."""
    documents = load_document_metadata()

    return [
        document
        for document in documents
        if query_matches_document(question, document)
        and not can_access_document(document, role, department)
    ]


def find_accessible_matching_documents(
    question: str,
    role: str,
    department: str,
    department_filter: str | None,
    file_type_filter: str | None,
) -> list[dict]:
    """Find accessible documents that match the question and selected filters."""
    documents = load_document_metadata()

    return [
        document
        for document in documents
        if query_matches_document(question, document)
        and can_access_document(document, role, department)
        and document_matches_filters(
            document,
            department_filter,
            file_type_filter,
        )
    ]


def calculate_chunk_evidence_score(chunk) -> int:
    """Rank chunks with concrete lists and steps ahead of introductory text."""
    content = chunk.page_content.lower()

    return (
        content.count("\n-")
        + content.count("step ")
        + content.count("must ")
    )


def build_context_and_sources(chunks:list) -> tuple[str,list]:
    """Convert retrieved LangChain Document chunks into plain text context for LLM and extract source filenames for citation."""
    #store text content for each retrieved chunk
    context_parts = []

    #store source filenames for citation display
    sources = []

    ordered_chunks = sorted(
        chunks,
        key=calculate_chunk_evidence_score,
        reverse=True,
    )

    for index, chunk in enumerate(ordered_chunks, start=1):
        source_name = chunk.metadata.get("source","Unknown source")
        sources.append(source_name)

        labelled_chunk = (
            f"[Source {index}: {source_name}]\n"
            f"{chunk.page_content}"
        )
        context_parts.append(labelled_chunk)

    #merge chunks into one prompt context block
    context_text = "\n\n".join(context_parts)

    #remove duplicates while preserving order
    unique_sources = list(dict.fromkeys(sources))

    #return in tuple
    return context_text,unique_sources


def generate_answer(
    question: str,
    role: str = "General Employee",
    department: str = "General",
    department_filter: str | None = None,
    file_type_filter: str | None = None,
) -> dict:
    """Generate a cited answer using only documents allowed for the user."""
    restricted_documents = find_restricted_matching_documents(
        question,
        role,
        department,
    )

    accessible_documents = find_accessible_matching_documents(
        question,
        role,
        department,
        department_filter,
        file_type_filter,
    )

    if restricted_documents and not accessible_documents:
        return {
            "question": question,
            "answer": (
                "Insufficient Permission. A relevant document exists, but your "
                "current role and department are not allowed to access that source."
            ),
            "sources": [],
        }
    
    matching_documents = [
        document
        for document in load_document_metadata()
        if query_matches_document(question, document)
    ]

    if matching_documents and not accessible_documents:
        return {
            "question": question,
            "answer": (
                "I could not find that information within the documents available "
                "for your current role, department, and selected filters."
            ),
            "sources": [],
        }

    #retrieve most relevant document chunks
    chunks = retrieve_relevant_chunks(
        question,
        role,
        department,
        department_filter,
        file_type_filter,
    )

    if not chunks:
        return {
            "question": question,
            "answer": (
                "I could not find relevant information in the documents available "
                "to your current role and department."
            ),
            "sources": [],
        }

    #prepare context and citation
    context_text, sources = build_context_and_sources(chunks)

    #system prompt (grounded rules for anti hallucination)
    prompt = f"""
    You are an internal knowledge assistant for Centific Malaysia.

    Answer the user's question using only the provided source excerpts.
    Use all relevant excerpts before deciding that information is missing.
    If any excerpt contains an explicit list of requirements, steps, or rules,
    copy the concrete items into the answer clearly and completely.
    Read every excerpt before deciding information is missing.
    Do not say details are missing when a later excerpt contains them.
    If the exact answer is genuinely not supported by any excerpt, say that
    the information was not found in the available documents.
    Do not infer policies, requirements, numbers, or rules from related but incomplete text.
    Temporary password setup instructions are not the same as password policy requirements.
    Extract facts exactly as written in the excerpts.
    Preserve every number exactly. Never replace a number with a different value.
    Do not merge, reinterpret, or add requirements from general knowledge.
    When an excerpt contains a bullet list, reproduce only those bullet items.
    Before answering, verify that every stated rule appears explicitly in the excerpts.

    Source excerpts:
    {context_text}

    Question:
    {question}
    """.strip()

    llm = Ollama(
        base_url= os.getenv("OLLAMA_BASE_URL"),
        model = os.getenv("OLLAMA_MODEL"),
        temperature=0,
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