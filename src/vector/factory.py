from src.core.config import (
    AZURE_SEARCH_VECTOR_BACKEND,
    CHROMA_VECTOR_BACKEND,
    AppConfig,
    read_app_config,
)
from src.vector import azure_search_backend, chroma_backend


def get_vector_backend_for_config(config: AppConfig):
    """Return the vector backend adapter for a specific app config."""
    if config.vector_backend == CHROMA_VECTOR_BACKEND:
        return chroma_backend

    if config.vector_backend == AZURE_SEARCH_VECTOR_BACKEND:
        return azure_search_backend

    raise ValueError(
        f"Unsupported VECTOR_BACKEND={config.vector_backend}. "
        "Supported values are 'chroma' and 'azure_search'."
    )


def get_vector_backend():
    """Return the active configured vector backend adapter."""
    return get_vector_backend_for_config(read_app_config())