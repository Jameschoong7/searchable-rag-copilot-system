import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.core.config import (
    AZURE_BLOB_STORAGE_BACKEND,
    LOCAL_STORAGE_BACKEND,
    read_app_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SIMULATED_DATA_PATH = PROJECT_ROOT / "data/simulated"


@dataclass(frozen=True)
class StoredDocument:
    """Describe where an uploaded document was stored."""

    filename: str
    local_path: str
    storage_backend: str
    storage_uri: str


def normalise_storage_filename(filename: str) -> str:
    """Return a safe local/blob filename for uploaded documents."""
    basename = Path(filename.replace("\\", "/")).name
    normalised = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return normalised or "uploaded_document"


def save_local_document_bytes(filename: str, content: bytes) -> StoredDocument:
    """Save document bytes into the local simulated document folder."""
    safe_filename = normalise_storage_filename(filename)
    LOCAL_SIMULATED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    local_path = LOCAL_SIMULATED_DATA_PATH / safe_filename
    local_path.write_bytes(content)

    return StoredDocument(
        filename=safe_filename,
        local_path=str(local_path),
        storage_backend=LOCAL_STORAGE_BACKEND,
        storage_uri=str(local_path),
    )


def upload_to_azure_blob(filename: str, content: bytes) -> str:
    """Upload document bytes to Azure Blob Storage and return the blob URL."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as error:
        raise RuntimeError(
            "azure-storage-blob is required when STORAGE_BACKEND=azure_blob."
        ) from error
    
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

    if not connection_string or not container_name:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER_NAME "
            "are required when STORAGE_BACKEND=azure_blob."
        )

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    try:
        container_client.create_container()
    except Exception:
        pass

    blob_client = container_client.get_blob_client(filename)
    blob_client.upload_blob(content, overwrite=True)

    return blob_client.url


def save_document_bytes(filename: str, content: bytes) -> StoredDocument:
    """Save document bytes using the configured storage backend."""
    app_config = read_app_config()
    local_copy = save_local_document_bytes(filename, content)

    if app_config.storage_backend == LOCAL_STORAGE_BACKEND:
        return local_copy

    if app_config.storage_backend == AZURE_BLOB_STORAGE_BACKEND:
        blob_uri = upload_to_azure_blob(local_copy.filename, content)

        return StoredDocument(
            filename=local_copy.filename,
            local_path=local_copy.local_path,
            storage_backend=AZURE_BLOB_STORAGE_BACKEND,
            storage_uri=blob_uri,
        )

    raise ValueError(f"Unsupported STORAGE_BACKEND: {app_config.storage_backend}")
