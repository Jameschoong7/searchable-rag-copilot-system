import os

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


def clear_vector_store_cache() -> None:
    """Azure Search client is stateless for this phase, so no local cache is cleared."""
    return None


def similarity_search_with_scores(question: str, allowed_sources: list[str], top_k: int) -> list[tuple]:
    raise NotImplementedError("Azure AI Search retrieval is not implemented yet.")


def store_chunks(chunks: list, db_path: str, collection_name: str):
    raise NotImplementedError("Azure AI Search full indexing is not implemented yet.")


def add_chunks(chunks: list, db_path: str, collection_name: str) -> None:
    raise NotImplementedError("Azure AI Search incremental indexing is not implemented yet.")


def delete_vectors_for_source(source_path: str, db_path: str, collection_name: str) -> int:
    raise NotImplementedError("Azure AI Search vector deletion is not implemented yet.")