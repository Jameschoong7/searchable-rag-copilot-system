from src.core.config import CHROMA_VECTOR_BACKEND, read_app_config
from src.vector import chroma_backend


def get_vector_backend():
    """Return the configured vector backend adapter."""
    config = read_app_config()

    if config.vector_backend == CHROMA_VECTOR_BACKEND:
        return chroma_backend

    raise ValueError(
        f"Unsupported VECTOR_BACKEND={config.vector_backend}. "
        "Only 'chroma' is implemented in this phase."
    )