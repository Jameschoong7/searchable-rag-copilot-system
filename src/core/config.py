import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)



LOCAL_STORAGE_BACKEND = "local"
AZURE_BLOB_STORAGE_BACKEND = "azure_blob"

CHROMA_VECTOR_BACKEND = "chroma"
AZURE_SEARCH_VECTOR_BACKEND = "azure_search"

OLLAMA_LLM_BACKEND = "ollama"
AZURE_OPENAI_LLM_BACKEND = "azure_openai"

SIMULATED_SHAREPOINT_MODE = "simulated"


@dataclass(frozen=True)
class AppConfig:
    """Central runtime configuration for local/Azure migration switches."""

    storage_backend: str
    vector_backend: str
    llm_backend: str
    graph_connector_enabled: bool
    sharepoint_mode: str


def read_bool_env(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable using simple true/false text."""
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def read_app_config() -> AppConfig:
    """Read backend mode switches from environment variables."""
    return AppConfig(
        storage_backend=os.getenv("STORAGE_BACKEND", LOCAL_STORAGE_BACKEND),
        vector_backend=os.getenv("VECTOR_BACKEND", CHROMA_VECTOR_BACKEND),
        llm_backend=os.getenv("LLM_BACKEND", OLLAMA_LLM_BACKEND),
        graph_connector_enabled=read_bool_env("GRAPH_CONNECTOR_ENABLED", False),
        sharepoint_mode=os.getenv("SHAREPOINT_MODE", SIMULATED_SHAREPOINT_MODE),
    )
