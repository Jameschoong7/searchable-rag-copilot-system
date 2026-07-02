from datetime import datetime
from pathlib import Path
import re

from src.core.constants import SYSTEM_ADMIN_ROLE, GENERAL_EMPLOYEE_ROLE
from src.metadata.repository import append_document_metadata
from src.storage.document_storage import save_document_bytes


DEPARTMENT_PATH_KEYWORDS = {
    "hr": "HR",
    "human resources": "HR",
    "it": "IT",
    "information technology": "IT",
    "engineering": "Engineering",
    "security": "Security",
    "operations": "Operations",
}


def stage_graph_file_for_review(
    document_id: str,
    title: str,
    original_filename: str,
    content_bytes: bytes,
    source_path: str,
    source_type: str,
    uploaded_by: str = "graph_connector",
) -> dict:
    """Stage a downloaded Graph file with a real local/Blob-backed source file."""
    stored_document = save_document_bytes(
        filename=original_filename,
        content=content_bytes,
    )

    metadata = build_pending_review_metadata(
        document_id=document_id,
        title=title,
        filename=stored_document.filename,
        source_path=source_path,
        source_type=source_type,
        uploaded_by=uploaded_by,
    )

    metadata["storage_backend"] = stored_document.storage_backend
    metadata["storage_uri"] = stored_document.storage_uri

    append_document_metadata(metadata)

    return metadata


def infer_department_from_source_path(source_path: str) -> str | None:
    """Infer a supported department from a OneDrive or OneNote source path."""
    normalized_path = source_path.lower()

    for keyword, department in DEPARTMENT_PATH_KEYWORDS.items():
        path_parts = [
            part.strip()
            for part in normalized_path.replace("\\", "/").split("/")
            if part.strip()
        ]

        if keyword in path_parts:
            return department

    return None


def build_pending_review_metadata(
    document_id: str,
    title: str,
    filename: str,
    source_path: str,
    source_type: str,
    uploaded_by: str = "graph_connector",
) -> dict:
    """Build safe pending-review metadata for a read-only Graph-ingested document."""
    inferred_department = infer_department_from_source_path(source_path)

    if inferred_department:
        department = inferred_department
        allowed_roles = [SYSTEM_ADMIN_ROLE, GENERAL_EMPLOYEE_ROLE]
        allowed_departments = [inferred_department]
    else:
        department = "IT"
        allowed_roles = [SYSTEM_ADMIN_ROLE]
        allowed_departments = ["IT"]

    return {
        "document_id": document_id,
        "title": title,
        "filename": filename,
        "file_type": Path(filename).suffix.replace(".", "").upper() or "TXT",
        "source": source_type,
        "department": department,
        "category": "Connector Import",
        "tags": ["graph", source_type, "pending-review"],
        "allowed_roles": allowed_roles,
        "allowed_departments": allowed_departments,
        "uploaded_by": uploaded_by,
        "uploaded_at": datetime.now().isoformat(timespec="minutes"),
        "page_number": None,
        "chunk_id": "pending_review",
        "visual_extraction_status": "Pending review",
        "source_document_id": document_id,
        "version_number": 1,
        "is_active": 0,
        "content_hash": None,
        "archived_at": None,
        "replaced_by_document_id": None,
        "storage_backend": "pending",
        "storage_uri": source_path,
    }


def stage_graph_document_for_review(
    document_id: str,
    title: str,
    filename: str,
    source_path: str,
    source_type: str,
    uploaded_by: str = "graph_connector",
) -> dict:
    """Deprecated metadata-only staging helper. Do not use for real connector ingestion."""
    metadata = build_pending_review_metadata(
        document_id=document_id,
        title=title,
        filename=filename,
        source_path=source_path,
        source_type=source_type,
        uploaded_by=uploaded_by,
    )

    append_document_metadata(metadata)

    return metadata


def build_graph_document_id(prefix: str, item_id: str) -> str:
    """Build a stable metadata ID from a Graph item ID."""
    safe_item_id = re.sub(r"[^A-Za-z0-9_-]+", "-", item_id).strip("-")
    return f"{prefix}-{safe_item_id}"


def build_graph_storage_filename(source_type: str, item_id: str, original_filename: str) -> str:
    """Build a unique stored filename while preserving the original extension."""
    original_path = Path(original_filename)
    safe_item_id = re.sub(r"[^A-Za-z0-9_-]+", "-", item_id).strip("-")
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", original_path.stem).strip("_")
    suffix = original_path.suffix

    return f"{source_type}_{safe_item_id}_{safe_stem}{suffix}"