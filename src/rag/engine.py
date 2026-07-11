#REQ_F004: Retrieve relevant documents chunks and generate cited answers
import os
import re
from pathlib import Path


from src.metadata.repository import load_document_metadata
from src.vector.factory import get_vector_backend
from src.core.config import read_app_config
from src.rag.llm_factory import create_chat_llm, invoke_configured_llm


from langchain_community.llms import Ollama
 # connects to the local Ollama server to run the Mistral model
  # ── AZURE SWAP ──
  # Replace Ollama with:
  #   from langchain_openai import AzureChatOpenAI
  # Requires: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
  # ── END AZURE SWAP ──


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        allowed_departments = document.get("allowed_departments", [])

        if (
            document["department"] != department_filter
            and department_filter not in allowed_departments
        ):
            return False

    if file_type_filter and file_type_filter != "All":
        if document["file_type"] != file_type_filter:
            return False

    return True


def query_matches_document(question: str, document: dict) -> bool:
    """Check whether a user question appears to target a document's metadata."""
    ignored_words = {
        "and",
        "are",
        "for",
        "how",
        "the",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
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
        if len(word) >= 3 and word not in ignored_words
    }

    searchable_metadata = " ".join(
        [
            document["title"],
            document.get("filename", ""),
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

    allowed_sources = []

    for filename in allowed_filenames:
        allowed_sources.extend(
            [
                f"data/simulated/{filename}",
                f"./data/simulated/{filename}",
                str(PROJECT_ROOT / "data/simulated" / filename),
            ]
        )

    return allowed_sources


def retrieve_relevant_chunks_with_scores(
    question: str,
    role: str,
    department: str,
    department_filter: str | None = None,
    file_type_filter: str | None = None,
    top_k: int | None = None,
    minimum_relevance_score: float | None = None,
) -> list[tuple]:
    """Retrieve scored chunks from documents allowed for the user's role and department."""

    app_config = read_app_config()
    effective_top_k = top_k if top_k is not None else app_config.top_k
    effective_minimum_relevance_score = (
        minimum_relevance_score
        if minimum_relevance_score is not None
        else app_config.minimum_relevance_threshold
    )

    allowed_sources = get_allowed_source_path(
        role,
        department,
        department_filter,
        file_type_filter,
    )

    if not allowed_sources:
        return []

    vector_backend = get_vector_backend()

    scored_results = vector_backend.similarity_search_with_scores(
        question=question,
        allowed_sources=allowed_sources,
        top_k=effective_top_k,
    )

    return [
        (document, score)
        for document, score in scored_results
        if score >= effective_minimum_relevance_score
    ]


def retrieve_relevant_chunks(
    question: str,
    role: str,
    department: str,
    department_filter: str | None = None,
    file_type_filter: str | None = None,
    top_k: int | None = None,
    minimum_relevance_score: float | None = None,
) -> list:
    """Retrieve only chunks from documents allowed for the user's role and department."""
    scored_chunks = retrieve_relevant_chunks_with_scores(
        question=question,
        role=role,
        department=department,
        department_filter=department_filter,
        file_type_filter=file_type_filter,
        top_k=top_k,
        minimum_relevance_score=minimum_relevance_score,
    )

    return [
        document
        for document, score in scored_chunks
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


def is_low_information_query(question: str) -> bool:
    """Detect vague test/greeting queries that should not trigger vector retrieval."""
    tokens = re.findall(r"[a-z0-9]+", question.lower())

    low_information_terms = {
        "test",
        "testing",
        "hello",
        "hi",
        "ok",
        "okay",
        "help",
    }

    return len(tokens) == 1 and tokens[0] in low_information_terms


def generate_answer(
    question: str,
    role: str = "General Employee",
    department: str = "General",
    department_filter: str | None = None,
    file_type_filter: str | None = None,
) -> dict:
    """Generate a cited answer using only documents allowed for the user."""

    if is_low_information_query(question):
        return {
            "question": question,
            "answer": (
                "I could not find a specific knowledge-base request in your question. "
                "Please ask a more specific question, such as a policy name, process, "
                "department, or task you need help with."
            ),
            "sources": [],
        }

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
    
    app_config = read_app_config()

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

    Additional admin guardrail: {app_config.guardrail_prompt}

    Source excerpts:
    {context_text}

    Question:
    {question}
    """.strip()

    llm = create_chat_llm()

    #send grounded prompt to LLM and get response
    answer = invoke_configured_llm(llm, prompt, operation="rag_answer")

    return {
        "question": question,
        "answer":answer,
        "sources":sources
    }
