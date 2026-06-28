import os
import hashlib
from pathlib import Path

from azure.search.documents import SearchClient

from src.metadata.repository import load_document_metadata

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
)
from src.vector.chroma_backend import load_embedding_model
from azure.search.documents.models import VectorizedQuery
from langchain_core.documents import Document

# Import shared config so direct module tests also load .env once.
from src.core.config import PROJECT_ROOT



VECTOR_DIMENSIONS = 384
VECTOR_PROFILE_NAME = "rag-vector-profile"
VECTOR_ALGORITHM_NAME = "rag-hnsw"


def get_azure_search_config() -> dict:
    """Read Azure AI Search settings from environment variables."""
    return {
        "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "admin_key": os.getenv("AZURE_SEARCH_ADMIN_KEY"),
        "index_name": os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-copilot-documents"),
    }


def validate_azure_search_config(config: dict) -> None:
    """Fail fast when Azure Search settings are missing."""
    missing = [
        key
        for key in ("endpoint", "admin_key", "index_name")
        if not config.get(key)
    ]

    if missing:
        raise ValueError(
            "Missing Azure AI Search configuration: "
            + ", ".join(missing)
        )


def get_index_client() -> SearchIndexClient:
    """Create an Azure AI Search index management client."""
    config = get_azure_search_config()
    validate_azure_search_config(config)

    return SearchIndexClient(
        endpoint=config["endpoint"],
        credential=AzureKeyCredential(config["admin_key"]),
    )


def build_search_index(index_name: str) -> SearchIndex:
    """Build the Azure AI Search schema for RAG chunks."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_document_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="department", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="file_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="is_active", type=SearchFieldDataType.Boolean, filterable=True),
        SimpleField(name="version_number", type=SearchFieldDataType.Int32, filterable=True),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name=VECTOR_ALGORITHM_NAME),
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=VECTOR_ALGORITHM_NAME,
            ),
        ],
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
    )


def create_or_update_index() -> str:
    """Create or update the configured Azure AI Search index."""
    config = get_azure_search_config()
    validate_azure_search_config(config)

    index_client = get_index_client()
    index = build_search_index(config["index_name"])
    index_client.create_or_update_index(index)

    return config["index_name"]


def get_search_client() -> SearchClient:
    """Create an Azure AI Search document upload/query client."""
    config = get_azure_search_config()
    validate_azure_search_config(config)

    return SearchClient(
        endpoint=config["endpoint"],
        index_name=config["index_name"],
        credential=AzureKeyCredential(config["admin_key"]),
    )


def load_metadata_by_filename() -> dict[str, dict]:
    """Map filenames to active SQLite metadata rows for Azure chunk enrichment."""
    return {
        document["filename"]: document
        for document in load_document_metadata()
    }


def build_chunk_id(source_path: str, chunk_index: int) -> str:
    """Create a stable Azure Search key for one source chunk."""
    raw_id = f"{source_path}:{chunk_index}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def normalize_bool(value) -> bool:
    """Convert SQLite-style active flags into real JSON booleans."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}

    return bool(value)


def build_search_document(chunk, chunk_index: int, metadata_by_filename: dict[str, dict]) -> dict:
    """Convert one LangChain chunk into an Azure AI Search document."""
    source_path = chunk.metadata.get("source", "")
    filename = Path(source_path).name
    metadata = metadata_by_filename.get(filename, {})

    return {
        "id": build_chunk_id(source_path, chunk_index),
        "content": chunk.page_content,
        "embedding": load_embedding_model().embed_query(chunk.page_content),
        "source": source_path,
        "filename": filename,
        "source_document_id": metadata.get("source_document_id"),
        "document_id": metadata.get("document_id"),
        "department": metadata.get("department"),
        "file_type": metadata.get("file_type"),
        "source_type": metadata.get("source_type"),
        "is_active": normalize_bool(metadata.get("is_active", True)),
        "version_number": int(metadata.get("version_number") or 1),
    }


def clear_vector_store_cache() -> None:
    """Azure Search client is stateless for this phase, so no local cache is cleared."""
    return None


def similarity_search_with_scores(question: str, allowed_sources: list[str], top_k: int) -> list[tuple]:
    """Run Azure AI Search vector retrieval inside the authorized source scope."""
    if not allowed_sources:
        return []

    query_vector = load_embedding_model().embed_query(question)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    search_client = get_search_client()
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        filter=build_allowed_sources_filter(allowed_sources),
        select=[
            "content",
            "source",
            "filename",
            "source_document_id",
            "document_id",
            "department",
            "file_type",
            "source_type",
            "version_number",
        ],
        top=top_k,
    )

    scored_documents = []

    for result in results:
        metadata = {
            "source": result.get("source"),
            "filename": result.get("filename"),
            "source_document_id": result.get("source_document_id"),
            "document_id": result.get("document_id"),
            "department": result.get("department"),
            "file_type": result.get("file_type"),
            "source_type": result.get("source_type"),
            "version_number": result.get("version_number"),
        }

        document = Document(
            page_content=result["content"],
            metadata=metadata,
        )

        score = float(result.get("@search.score", 0))
        scored_documents.append((document, score))

    return scored_documents


def store_chunks(chunks: list, db_path: str, collection_name: str):
    """Create/update index and upload chunks into Azure AI Search."""
    add_chunks(chunks, db_path, collection_name)
    print(f"Uploaded {len(chunks)} chunks to Azure AI Search")
    return None


def add_chunks(chunks: list, db_path: str, collection_name: str) -> None:
    """Upload chunked documents into Azure AI Search."""
    if not chunks:
        return None

    create_or_update_index()

    metadata_by_filename = load_metadata_by_filename()
    documents = [
        build_search_document(chunk, chunk_index, metadata_by_filename)
        for chunk_index, chunk in enumerate(chunks)
    ]

    search_client = get_search_client()
    result = search_client.upload_documents(documents)

    failed = [
        item
        for item in result
        if not item.succeeded
    ]

    if failed:
        raise RuntimeError(f"Azure AI Search failed to upload {len(failed)} chunk(s).")

    return None


def delete_vectors_for_source(source_path: str, db_path: str, collection_name: str) -> int:
    """Delete Azure AI Search chunks that came from one source file."""
    search_client = get_search_client()
    escaped_source = escape_odata_string(source_path)

    results = search_client.search(
        search_text="*",
        filter=f"source eq '{escaped_source}'",
        select=["id"],
        top=1000,
    )

    documents_to_delete = [
        {"id": result["id"]}
        for result in results
    ]

    if not documents_to_delete:
        return 0

    delete_results = search_client.delete_documents(documents_to_delete)

    failed = [
        item
        for item in delete_results
        if not item.succeeded
    ]

    if failed:
        raise RuntimeError(f"Azure AI Search failed to delete {len(failed)} chunk(s).")

    return len(documents_to_delete)


def escape_odata_string(value: str) -> str:
    """Escape single quotes for Azure Search OData filter strings."""
    return value.replace("'", "''")


def build_allowed_sources_filter(allowed_sources: list[str]) -> str:
    """Build an Azure Search filter from backend-approved source paths."""
    escaped_sources = [
        escape_odata_string(source)
        for source in allowed_sources
    ]

    source_list = ",".join(escaped_sources)

    return f"is_active eq true and search.in(source, '{source_list}', ',')"


def reset_index(db_path: str, collection_name: str) -> None:
    """Reset the Azure AI Search index before a full rebuild."""
    config = get_azure_search_config()
    validate_azure_search_config(config)

    index_client = get_index_client()

    try:
        index_client.delete_index(config["index_name"])
    except Exception:
        pass

    create_or_update_index()


def get_index_record_count() -> int:
    """Count chunk records stored in the Azure AI Search index."""
    search_client = get_search_client()
    return search_client.get_document_count()


def get_index_size_bytes() -> int | None:
    """Azure Search index size is not exposed by the data-plane SearchClient."""
    return None