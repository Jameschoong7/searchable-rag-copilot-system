import os


def get_azure_search_config() -> dict:
    """Read Azure AI Search settings from environment variables."""
    return {
        "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "admin_key": os.getenv("AZURE_SEARCH_ADMIN_KEY"),
        "index_name": os.getenv("AZURE_SEARCH_INDEX_NAME", "ragcopilot-aisearch"),
    }


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