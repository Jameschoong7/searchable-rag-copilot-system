import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.core.settings_repository import (
    load_pending_runtime_settings,
    load_runtime_settings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


LOCAL_STORAGE_BACKEND = "local"
AZURE_BLOB_STORAGE_BACKEND = "azure_blob"

CHROMA_VECTOR_BACKEND = "chroma"
AZURE_SEARCH_VECTOR_BACKEND = "azure_search"

LOCAL_EMBEDDING_BACKEND = "local"
AZURE_OPENAI_EMBEDDING_BACKEND = "azure_openai"

OLLAMA_LLM_BACKEND = "ollama"
AZURE_OPENAI_LLM_BACKEND = "azure_openai"

SIMULATED_SHAREPOINT_MODE = "simulated"

DEFAULT_TOP_K = 5
DEFAULT_MINIMUM_RELEVANCE_THRESHOLD = 0.25
DEFAULT_GUARDRAIL_PROMPT = (
    "Answer only from authorised retrieved documents. "
    "If the answer is missing or restricted, say so clearly."
)


@dataclass(frozen=True)
class AppConfig:
    """Central runtime configuration for local/Azure migration switches."""

    storage_backend: str
    vector_backend: str
    embedding_backend: str
    llm_backend: str
    graph_connector_enabled: bool
    graph_access_token: str | None
    graph_onedrive_root_path: str
    graph_onenote_notebook_filter: str | None
    sharepoint_mode: str
    top_k: int
    minimum_relevance_threshold: float
    guardrail_prompt: str


def read_bool_value(value: str | None, default: bool = False) -> bool:
    """Convert simple true/false text into a boolean."""
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def read_config_value(
    key: str,
    default: str,
    runtime_settings: dict[str, str],
) -> str:
    """Read one setting from SQLite overrides first, then environment defaults."""
    return runtime_settings.get(
        key.lower(),
        os.getenv(key, default),
    )


def read_app_config() -> AppConfig:
    """Read backend mode switches from environment defaults plus SQLite overrides."""
    runtime_settings = load_runtime_settings()

    return AppConfig(
        storage_backend=read_config_value(
            "STORAGE_BACKEND",
            LOCAL_STORAGE_BACKEND,
            runtime_settings,
        ),
        vector_backend=read_config_value(
            "VECTOR_BACKEND",
            CHROMA_VECTOR_BACKEND,
            runtime_settings,
        ),
        embedding_backend=read_config_value(
            "EMBEDDING_BACKEND",
            LOCAL_EMBEDDING_BACKEND,
            runtime_settings,
        ),
        llm_backend=read_config_value(
            "LLM_BACKEND",
            OLLAMA_LLM_BACKEND,
            runtime_settings,
        ),
        graph_connector_enabled=read_bool_value(
            read_config_value(
                "GRAPH_CONNECTOR_ENABLED",
                "false",
                runtime_settings,
            )
        ),
        graph_access_token=os.getenv("GRAPH_ACCESS_TOKEN"),
        graph_onedrive_root_path=os.getenv(
            "GRAPH_ONEDRIVE_ROOT_PATH",
            "/Enterprise Knowledge Base",
        ),
        graph_onenote_notebook_filter=os.getenv("GRAPH_ONENOTE_NOTEBOOK_FILTER"),
        sharepoint_mode=read_config_value(
            "SHAREPOINT_MODE",
            SIMULATED_SHAREPOINT_MODE,
            runtime_settings,
        ),
        top_k=int(
            read_config_value(
                "TOP_K",
                str(DEFAULT_TOP_K),
                runtime_settings,
            )
        ),
        minimum_relevance_threshold=float(
            read_config_value(
                "MINIMUM_RELEVANCE_THRESHOLD",
                str(DEFAULT_MINIMUM_RELEVANCE_THRESHOLD),
                runtime_settings,
            )
        ),
        guardrail_prompt=read_config_value(
            "GUARDRAIL_PROMPT",
            DEFAULT_GUARDRAIL_PROMPT,
            runtime_settings,
        ),
    )


def read_app_config_with_pending() -> AppConfig:
    """Read active config with pending risky settings overlaid for rebuild targets."""
    runtime_settings = load_runtime_settings()
    pending_settings = load_pending_runtime_settings()

    merged_settings = {
        **runtime_settings,
        **pending_settings,
    }

    return AppConfig(
        storage_backend=read_config_value(
            "STORAGE_BACKEND",
            LOCAL_STORAGE_BACKEND,
            merged_settings,
        ),
        vector_backend=read_config_value(
            "VECTOR_BACKEND",
            CHROMA_VECTOR_BACKEND,
            merged_settings,
        ),
        embedding_backend=read_config_value(
            "EMBEDDING_BACKEND",
            LOCAL_EMBEDDING_BACKEND,
            merged_settings,
        ),
        llm_backend=read_config_value(
            "LLM_BACKEND",
            OLLAMA_LLM_BACKEND,
            merged_settings,
        ),
        graph_connector_enabled=read_bool_value(
            read_config_value(
                "GRAPH_CONNECTOR_ENABLED",
                "false",
                merged_settings,
            )
        ),
        graph_access_token=os.getenv("GRAPH_ACCESS_TOKEN"),
        graph_onedrive_root_path=os.getenv(
            "GRAPH_ONEDRIVE_ROOT_PATH",
            "/Enterprise Knowledge Base",
        ),
        graph_onenote_notebook_filter=os.getenv("GRAPH_ONENOTE_NOTEBOOK_FILTER"),
        sharepoint_mode=read_config_value(
            "SHAREPOINT_MODE",
            SIMULATED_SHAREPOINT_MODE,
            merged_settings,
        ),
        top_k=int(
            read_config_value(
                "TOP_K",
                str(DEFAULT_TOP_K),
                merged_settings,
            )
        ),
        minimum_relevance_threshold=float(
            read_config_value(
                "MINIMUM_RELEVANCE_THRESHOLD",
                str(DEFAULT_MINIMUM_RELEVANCE_THRESHOLD),
                merged_settings,
            )
        ),
        guardrail_prompt=read_config_value(
            "GUARDRAIL_PROMPT",
            DEFAULT_GUARDRAIL_PROMPT,
            merged_settings,
        ),
    )


SETTING_STORAGE_BACKEND = "storage_backend"
SETTING_VECTOR_BACKEND = "vector_backend"
SETTING_EMBEDDING_BACKEND = "embedding_backend"
SETTING_LLM_BACKEND = "llm_backend"
SETTING_TOP_K = "top_k"
SETTING_MINIMUM_RELEVANCE_THRESHOLD = "minimum_relevance_threshold"
SETTING_GUARDRAIL_PROMPT = "guardrail_prompt"

REBUILD_REQUIRED_SETTINGS = {
    SETTING_VECTOR_BACKEND,
    SETTING_EMBEDDING_BACKEND,
}

ALLOWED_SETTING_VALUES = {
    SETTING_STORAGE_BACKEND: {
        LOCAL_STORAGE_BACKEND,
        AZURE_BLOB_STORAGE_BACKEND,
    },
    SETTING_VECTOR_BACKEND: {
        CHROMA_VECTOR_BACKEND,
        AZURE_SEARCH_VECTOR_BACKEND,
    },
    SETTING_EMBEDDING_BACKEND: {
        LOCAL_EMBEDDING_BACKEND,
        AZURE_OPENAI_EMBEDDING_BACKEND,
    },
    SETTING_LLM_BACKEND: {
        OLLAMA_LLM_BACKEND,
        AZURE_OPENAI_LLM_BACKEND,
    },
}


def get_config_as_settings_dict(config: AppConfig) -> dict[str, str]:
    """Convert the active app config into the editable settings shape."""
    return {
        SETTING_STORAGE_BACKEND: config.storage_backend,
        SETTING_VECTOR_BACKEND: config.vector_backend,
        SETTING_EMBEDDING_BACKEND: config.embedding_backend,
        SETTING_LLM_BACKEND: config.llm_backend,
        SETTING_TOP_K: str(config.top_k),
        SETTING_MINIMUM_RELEVANCE_THRESHOLD: str(config.minimum_relevance_threshold),
        SETTING_GUARDRAIL_PROMPT: config.guardrail_prompt,
    }


def validate_runtime_settings(settings: dict[str, str]) -> None:
    """Validate admin-provided runtime settings before saving them."""
    for key, allowed_values in ALLOWED_SETTING_VALUES.items():
        if key in settings and settings[key] not in allowed_values:
            raise ValueError(
                f"Invalid {key}: {settings[key]}. "
                f"Allowed values: {', '.join(sorted(allowed_values))}"
            )

    if SETTING_TOP_K in settings:
        top_k = int(settings[SETTING_TOP_K])

        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20.")

    if SETTING_MINIMUM_RELEVANCE_THRESHOLD in settings:
        threshold = float(settings[SETTING_MINIMUM_RELEVANCE_THRESHOLD])

        if threshold < 0 or threshold > 1:
            raise ValueError("minimum_relevance_threshold must be between 0 and 1.")

    if SETTING_GUARDRAIL_PROMPT in settings:
        if not settings[SETTING_GUARDRAIL_PROMPT].strip():
            raise ValueError("guardrail_prompt cannot be empty.")


def find_changed_settings(
    current_settings: dict[str, str],
    new_settings: dict[str, str],
) -> list[str]:
    """Return setting keys whose values changed."""
    return [
        key
        for key, new_value in new_settings.items()
        if str(current_settings.get(key)) != str(new_value)
    ]


def settings_require_rebuild(changed_keys: list[str]) -> bool:
    """Check whether changed settings require a full search index rebuild."""
    return any(
        key in REBUILD_REQUIRED_SETTINGS
        for key in changed_keys
    )