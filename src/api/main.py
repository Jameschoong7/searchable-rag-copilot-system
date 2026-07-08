# REQ_F004: Exposes cited RAG answers through an API endpoint
# REQ_F003: Provides the backend route that a future Teams chatbot can call
# REQ_F005: Provides the backend route that the Streamlit web app can call

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
import re
from datetime import datetime
import hashlib
import json
import zipfile
import io
from pathlib import Path


# Create the shared FastAPI application used by both frontend platforms.
app = FastAPI(
    title="Searchable RAG Copilot API",
    description="Shared backend brain for the Admin Web Portal and simulated MS Teams Chatbot.",
    version="0.1.0",
)

from src.core.constants import (
    GENERAL_EMPLOYEE_ROLE,
    PROJECT_MANAGER_ROLE,
    SYSTEM_ADMIN_ROLE,
    expand_allowed_departments,
    expand_allowed_roles,
)


from src.core.config import (
    find_changed_settings,
    get_config_as_settings_dict,
    read_app_config,
    read_app_config_with_pending,
    settings_require_rebuild,
    validate_runtime_settings,
)
from src.core.settings_repository import (
    load_pending_runtime_settings,
    promote_pending_runtime_settings,
    save_pending_runtime_settings,
    save_runtime_settings,
)


from src.core.job_repository import (
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
    JOB_TYPE_CHAT_QUERY,
    JOB_TYPE_REINDEX,
    JOB_TYPE_INDEX_UPDATE,
    JOB_TYPE_ONEDRIVE_STAGE,
    JOB_TYPE_ONEDRIVE_REFRESH,
    JOB_TYPE_ONENOTE_STAGE,
    JOB_TYPE_ONENOTE_REFRESH,
    JOB_TYPE_DOCUMENT_ARCHIVE,
    JOB_TYPE_DOCUMENT_UNARCHIVE,
    JOB_TYPE_INDEX_SNAPSHOT,
    create_job,
    get_job,
    get_latest_job,
    update_job,
)
from src.core.answer_status import classify_answer_status_detail
from src.core.chat_memory_repository import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_USER,
    append_chat_message,
    get_or_create_chat_session,
    list_chat_messages_for_session,
    list_recent_chat_messages_for_session,
    list_chat_sessions_for_user,
)
from src.rag.chat_rewrite import (
    MAX_REWRITE_HISTORY_MESSAGES,
    rewrite_follow_up_question_with_configured_llm,
)


def is_meaningful_question(question: str) -> bool:
    """Reject punctuation-only or too-short queries before retrieval."""
    words = re.findall(r"[a-zA-Z0-9]+", question)

    if not words:
        return False

    meaningful_words = [
        word for word in words
        if len(word) >= 2
    ]

    return bool(meaningful_words)


@app.get("/health")
def health_check() -> dict:
    """Return a simple status check so clients can confirm the API is running."""
    return {
        "status": "ok",
        "service": "Searchable RAG Copilot API",
    }


class QueryRequest(BaseModel):
    """Represent one user question sent from a frontend client."""

    question: str
    role: str = GENERAL_EMPLOYEE_ROLE
    department: str = "General"
    department_filter: str | None = None
    file_type_filter: str | None = None


class QueryResponse(BaseModel):
    """Represent the cited answer returned by the shared RAG backend."""

    question: str
    answer: str
    sources: list[str]
    role: str
    department: str


class GraphConnectorListRequest(BaseModel):
    """Represent an admin request to list files from the configured OneDrive root."""

    role: str


class OneDriveFileSummary(BaseModel):
    """Represent one file discovered from the configured OneDrive connector root."""

    id: str
    name: str
    connector_path: str
    size: int | None = None
    last_modified_datetime: str | None = None
    connector_state: str = "New"
    staged_document_id: str | None = None


class OneDriveFilesResponse(BaseModel):
    """Represent files discovered from the configured OneDrive connector root."""

    status: str
    files: list[OneDriveFileSummary]


class StageOneDriveFileRequest(BaseModel):
    """Represent an admin request to stage one OneDrive file for metadata review."""

    role: str
    user: str
    item_id: str
    name: str
    connector_path: str


class StageOneDriveFileResponse(BaseModel):
    """Represent a OneDrive file staged as a pending-review KB document."""

    status: str
    document_id: str
    filename: str
    storage_backend: str
    storage_uri: str
    chunk_id: str
    message: str


class OneDriveStageFileItem(BaseModel):
    """Represent one OneDrive file selected for batch staging."""

    item_id: str
    name: str
    connector_path: str


class StageOneDriveFilesJobRequest(BaseModel):
    """Represent a durable batch OneDrive staging request."""

    role: str
    user: str
    files: list[OneDriveStageFileItem]


class RefreshOneDriveFileRequest(BaseModel):
    """Represent an admin request to refresh one already-ingested OneDrive file."""

    role: str
    user: str
    user_department: str
    item_id: str
    name: str
    connector_path: str


class RefreshOneDriveFileResponse(BaseModel):
    """Represent the result of refreshing one OneDrive-backed KB document."""

    status: str
    document_id: str | None = None
    previous_document_id: str | None = None
    chunk_id: str | None = None
    message: str


class OneDriveRefreshFileItem(BaseModel):
    """Represent one OneDrive file selected for batch refresh."""

    item_id: str
    name: str
    connector_path: str


class RefreshOneDriveFilesJobRequest(BaseModel):
    """Represent a durable batch OneDrive refresh request."""

    role: str
    user: str
    user_department: str
    files: list[OneDriveRefreshFileItem]


class OneNotePageSummary(BaseModel):
    """Represent one OneNote page discovered through Microsoft Graph."""

    id: str
    title: str
    notebook_name: str
    section_name: str
    connector_path: str
    last_modified_datetime: str | None = None
    connector_state: str = "New"
    staged_document_id: str | None = None


class OneNotePagesResponse(BaseModel):
    """Represent OneNote pages discovered through Microsoft Graph."""

    status: str
    pages: list[OneNotePageSummary]


class StageOneNotePageRequest(BaseModel):
    """Represent an admin request to stage one OneNote page for metadata review."""

    role: str
    user: str
    page_id: str
    title: str
    connector_path: str


class StageOneNotePageResponse(BaseModel):
    """Represent a OneNote page staged as a pending-review KB document."""

    status: str
    document_id: str
    filename: str
    storage_backend: str
    storage_uri: str
    chunk_id: str
    message: str


class RefreshOneNotePageRequest(BaseModel):
    """Represent an admin request to refresh one already-ingested OneNote page."""

    role: str
    user: str
    user_department: str
    page_id: str
    title: str
    connector_path: str


class RefreshOneNotePageResponse(BaseModel):
    """Represent the result of refreshing one OneNote-backed KB document."""

    status: str
    document_id: str | None = None
    previous_document_id: str | None = None
    chunk_id: str | None = None
    message: str


class OneNoteStagePageItem(BaseModel):
    """Represent one OneNote page selected for batch staging."""

    page_id: str
    title: str
    connector_path: str


class StageOneNotePagesJobRequest(BaseModel):
    """Represent a durable batch OneNote staging request."""

    role: str
    user: str
    pages: list[OneNoteStagePageItem]


class OneNoteRefreshPageItem(BaseModel):
    """Represent one OneNote page selected for batch refresh."""

    page_id: str
    title: str
    connector_path: str


class RefreshOneNotePagesJobRequest(BaseModel):
    """Represent a durable batch OneNote refresh request."""

    role: str
    user: str
    user_department: str
    pages: list[OneNoteRefreshPageItem]


class ReindexRequest(BaseModel):
    """Represent an admin reindex request from a frontend client."""

    role: str


class ReindexJobRequest(ReindexRequest):
    """Represent a durable reindex request submitted as a backend job."""

    user: str


class ReindexResponse(BaseModel):
    """Represent the result of a local vector index rebuild."""

    status: str
    documents_indexed: int
    document_objects_loaded: int
    chunks_indexed: int
    message: str


class IndexUpdatesRequest(BaseModel):
    """Represent an admin request to index pending document updates."""

    role: str


class IndexUpdatesJobRequest(IndexUpdatesRequest):
    """Represent a durable pending-index update request submitted as a backend job."""

    user: str


class IndexSnapshotJobRequest(BaseModel):
    """Represent an admin request to refresh the saved index snapshot."""

    role: str
    user: str


class IndexUpdatesResponse(BaseModel):
    """Represent the result of indexing pending document updates."""

    status: str
    pending_document_count: int
    updated_sources: list[str]
    total_deleted_vectors: int
    total_chunks_indexed: int
    elapsed_seconds: float
    message: str


class ArchiveDocumentRequest(BaseModel):
    """Represent an admin request to archive one active document."""

    role: str
    user_department: str
    document_id: str


class ArchiveDocumentJobRequest(ArchiveDocumentRequest):
    """Represent a durable archive request submitted as a backend job."""

    user: str


class ArchiveDocumentResponse(BaseModel):
    """Represent the result of archiving one document and deleting its vectors."""

    status: str
    document_id: str
    deleted_vector_count: int
    message: str


class UploadValidationRequest(BaseModel):
    """Represent metadata proposed for a local simulated document upload."""

    role: str
    user_department: str
    document_department: str
    allowed_roles: list[str]
    allowed_departments: list[str]


class UnarchiveDocumentRequest(BaseModel):
    """Represent an admin request to restore one manually archived document."""

    role: str
    user_department: str
    document_id: str


class UnarchiveDocumentJobRequest(UnarchiveDocumentRequest):
    """Represent a durable restore request submitted as a backend job."""

    user: str


class UnarchiveDocumentResponse(BaseModel):
    """Represent the result of restoring one archived document."""

    status: str
    document_id: str
    chunk_id: str
    message: str


class UploadValidationResponse(BaseModel):
    """Represent backend-approved upload metadata scope."""

    status: str
    document_department: str
    allowed_roles: list[str]
    allowed_departments: list[str]


class UploadDocumentResponse(BaseModel):
    """Represent a document created by a backend-owned upload."""

    status: str
    document_id: str
    filename: str
    storage_backend: str
    storage_uri: str
    chunk_id: str
    message: str


class UploadDocumentVersionResponse(UploadDocumentResponse):
    """Represent a replacement version created by a backend-owned upload."""

    previous_document_id: str
    version_number: int


class BatchZipStageResponse(BaseModel):
    """Represent the result of staging supported files from one ZIP upload."""

    status: str
    staged_count: int
    skipped_count: int
    results: list[dict]
    message: str


class MetadataUpdateValidationRequest(BaseModel):
    """Represent metadata proposed for an existing document update."""

    role: str
    user_department: str
    document_department: str
    allowed_roles: list[str]
    allowed_departments: list[str]


class MetadataUpdateValidationResponse(BaseModel):
    """Represent backend-approved metadata update scope."""

    status: str
    document_department: str
    allowed_roles: list[str]
    allowed_departments: list[str]


class SettingsResponse(BaseModel):
    """Represent current backend-owned runtime settings."""

    settings: dict[str, str]
    pending_settings: dict[str, str] = {}
    rebuild_required: bool = False
    changed_keys: list[str] = []
    message: str = ""


class SettingsUpdateRequest(BaseModel):
    """Represent admin-submitted runtime setting changes."""

    role: str
    updated_by: str
    settings: dict[str, str]


class SettingsUpdateResponse(BaseModel):
    """Represent the result of saving backend-owned runtime settings."""

    status: str
    settings: dict[str, str]
    pending_settings: dict[str, str]
    changed_keys: list[str]
    rebuild_required: bool
    message: str


class ChatJobRequest(QueryRequest):
    """Represent a durable chat query submitted as a backend job."""

    user: str
    session_id: str | None = None
    use_memory: bool = True


class JobResponse(BaseModel):
    """Represent one backend job record."""

    job_id: str
    job_type: str
    status: str
    message: str
    result: dict
    created_by: str
    created_at: str
    updated_at: str


class ChatSessionResponse(BaseModel):
    """Represent one persisted chat conversation session."""

    session_id: str
    user: str
    role: str
    department: str
    title: str
    created_at: str
    updated_at: str


class ChatMessageResponse(BaseModel):
    """Represent one persisted chat message."""

    message_id: str
    session_id: str
    message_role: str
    content: str
    sources: list[str]
    status: str
    created_at: str


class ApproveDocumentRequest(BaseModel):
    """Represent an admin request to approve a pending-review document."""

    role: str
    user_department: str
    document_id: str
    title: str
    department: str
    category: str
    tags: list[str]
    allowed_roles: list[str]
    allowed_departments: list[str]


class ApproveDocumentResponse(BaseModel):
    """Represent a connector document approved for indexing."""

    status: str
    document_id: str
    chunk_id: str
    message: str


class RejectStagedDocumentRequest(BaseModel):
    """Represent an admin request to reject a pending-review connector document."""

    role: str
    user_department: str
    document_id: str


def run_chat_query_job(job_id: str, request: ChatJobRequest) -> None:
    """Run one chat query in the background and store the answer in the job table."""
    import time

    question = request.question.strip()
    chat_session = get_or_create_chat_session(
        session_id=request.session_id,
        user=request.user,
        role=request.role,
        department=request.department,
        first_question=question,
    )
    session_id = chat_session["session_id"]
    retrieval_question = question

    if request.use_memory:
        recent_messages = list_recent_chat_messages_for_session(
            session_id=session_id,
            user=request.user,
            limit=MAX_REWRITE_HISTORY_MESSAGES,
        )
        retrieval_question = rewrite_follow_up_question_with_configured_llm(
            question,
            recent_messages,
        )

    append_chat_message(
        session_id=session_id,
        message_role=MESSAGE_ROLE_USER,
        content=question,
        sources=[],
        status="submitted",
    )

    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        "Retrieving authorised knowledge chunks and generating answer.",
    )

    start_time = time.perf_counter()

    try:
        from src.rag.engine import generate_answer

        result = generate_answer(
            question=retrieval_question,
            role=request.role,
            department=request.department,
            department_filter=request.department_filter,
            file_type_filter=request.file_type_filter,
        )
        answer_status_detail = classify_answer_status_detail(
            result["answer"],
            result["sources"],
        )
        answer_status = answer_status_detail["status"]

        append_chat_message(
            session_id=session_id,
            message_role=MESSAGE_ROLE_ASSISTANT,
            content=result["answer"],
            sources=result["sources"],
            status=answer_status,
        )

        update_job(
            job_id,
            JOB_STATUS_SUCCEEDED,
            "Chat answer generated.",
            {
                "session_id": session_id,
                "question": question,
                "retrieval_question": retrieval_question,
                "use_memory": request.use_memory,
                "answer": result["answer"],
                "sources": result["sources"],
                "answer_status": answer_status,
                "status_reason": answer_status_detail["reason"],
                "role": request.role,
                "department": request.department,
                "department_filter": request.department_filter,
                "file_type_filter": request.file_type_filter,
                "latency_seconds": round(time.perf_counter() - start_time, 3),
            },
        )
    except Exception as error:
        append_chat_message(
            session_id=session_id,
            message_role=MESSAGE_ROLE_ASSISTANT,
            content=f"Chat query failed: {error}",
            sources=[],
            status="api_error",
        )

        update_job(
            job_id,
            JOB_STATUS_FAILED,
            f"Chat query failed: {error}",
            {
                "session_id": session_id,
                "question": question,
                "retrieval_question": retrieval_question,
                "use_memory": request.use_memory,
                "answer": "",
                "sources": [],
                "answer_status": "api_error",
                "status_reason": "Backend chat job failed",
                "role": request.role,
                "department": request.department,
                "department_filter": request.department_filter,
                "file_type_filter": request.file_type_filter,
                "latency_seconds": round(time.perf_counter() - start_time, 3),
            },
        )


def run_reindex_job(job_id: str, request: ReindexJobRequest) -> None:
    """Run one full reindex in the background and store status in the job table."""
    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        "Rebuilding target search index. Active settings remain unchanged until success.",
    )

    try:
        result = run_full_reindex(updated_by=request.user)

        update_job(
            job_id,
            JOB_STATUS_SUCCEEDED,
            result["message"],
            result,
        )
    except Exception as error:
        update_job(
            job_id,
            JOB_STATUS_FAILED,
            f"Index rebuild failed. Active settings were not changed: {error}",
            {
                "status": "failed",
                "documents_indexed": 0,
                "document_objects_loaded": 0,
                "chunks_indexed": 0,
                "message": str(error),
            },
        )


def run_index_update_job(job_id: str, request: IndexUpdatesJobRequest) -> None:
    """Run pending document indexing in the background and store job status."""
    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        "Updating pending documents in the active search index.",
    )

    try:
        result = run_pending_index_update(updated_by=request.user)

        update_job(
            job_id,
            JOB_STATUS_SUCCEEDED,
            result["message"],
            result,
        )
    except Exception as error:
        update_job(
            job_id,
            JOB_STATUS_FAILED,
            f"Pending index update failed: {error}",
            {
                "status": "failed",
                "pending_document_count": 0,
                "updated_sources": [],
                "total_deleted_vectors": 0,
                "total_chunks_indexed": 0,
                "elapsed_seconds": 0,
                "message": str(error),
            },
        )


def run_index_snapshot_job(job_id: str, request: IndexSnapshotJobRequest) -> None:
    """Refresh the saved index snapshot without rebuilding or changing vectors."""
    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        "Refreshing saved index snapshot.",
    )

    try:
        from src.evaluation.index_benchmark import (
            build_index_benchmark_snapshot,
            save_benchmark_result,
        )

        snapshot = build_index_benchmark_snapshot()
        snapshot["benchmark_type"] = "snapshot"
        snapshot["updated_by"] = request.user
        snapshot["message"] = (
            f"Saved index snapshot with {snapshot['indexed_chunk_count']} indexed chunk(s)."
        )
        save_benchmark_result(snapshot)

        update_job(
            job_id,
            JOB_STATUS_SUCCEEDED,
            snapshot["message"],
            snapshot,
        )
    except Exception as error:
        update_job(
            job_id,
            JOB_STATUS_FAILED,
            f"Index snapshot refresh failed: {error}",
            {
                "status": "failed",
                "message": str(error),
            },
        )


def approve_upload_scope(
    role: str,
    user_department: str,
    document_department: str,
    allowed_roles: list[str],
    allowed_departments: list[str],
) -> UploadValidationResponse:
    """Apply backend upload scope rules before file or metadata writes."""
    if role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot upload knowledge base documents.",
        )

    if role == PROJECT_MANAGER_ROLE:
        return UploadValidationResponse(
            status="approved",
            document_department=user_department,
            allowed_roles=expand_allowed_roles(
                [
                    allowed_role for allowed_role in allowed_roles
                    if allowed_role in [PROJECT_MANAGER_ROLE, GENERAL_EMPLOYEE_ROLE]
                ] or [PROJECT_MANAGER_ROLE]
            ),
            allowed_departments=[user_department],
        )

    if role == SYSTEM_ADMIN_ROLE:
        if not allowed_roles:
            raise HTTPException(
                status_code=400,
                detail="At least one allowed role is required.",
            )

        if not allowed_departments:
            raise HTTPException(
                status_code=400,
                detail="At least one allowed department is required.",
            )

        return UploadValidationResponse(
            status="approved",
            document_department=document_department,
            allowed_roles=expand_allowed_roles(allowed_roles),
            allowed_departments=expand_allowed_departments(allowed_departments),
        )

    raise HTTPException(
        status_code=403,
        detail="Unknown role cannot upload knowledge base documents.",
    )


def parse_json_list_field(raw_value: str, field_name: str) -> list[str]:
    """Parse a JSON list submitted through multipart form data."""
    try:
        parsed_value = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a JSON list.",
        ) from error

    if not isinstance(parsed_value, list):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a JSON list.",
        )

    return [
        str(item).strip()
        for item in parsed_value
        if str(item).strip()
    ]


def get_uploaded_file_type(filename: str) -> str:
    """Return the metadata file type for a supported uploaded file."""
    lowered_filename = filename.lower()

    if lowered_filename.endswith(".txt"):
        return "TXT"

    if lowered_filename.endswith(".pdf"):
        return "PDF"

    if lowered_filename.endswith(".docx"):
        return "DOCX"

    raise HTTPException(
        status_code=400,
        detail="Only TXT, PDF, and DOCX uploads are supported.",
    )


def get_visual_extraction_status(file_type: str, file_bytes: bytes | None = None) -> str:
    """Return the extraction status label for a supported uploaded file type."""
    if file_type == "PDF":
        if file_bytes:
            try:
                from src.etl.visual_detection import detect_pdf_visual_status

                return detect_pdf_visual_status(file_bytes)
            except Exception:
                return "PDF text extraction - visual status unknown"

        return "PDF text extraction"

    if file_type == "DOCX":
        return "Word text extraction"

    return "Text only"


def generate_version_document_id(previous_document: dict, next_version_number: int) -> str:
    """Generate a readable document ID for a new version of an existing document."""
    source_document_id = previous_document.get(
        "source_document_id",
        previous_document["document_id"],
    )

    return f"{source_document_id}-V{next_version_number}"


def build_versioned_filename(
    previous_document: dict,
    original_filename: str,
    next_version_number: int,
) -> str:
    """Build a unique stored filename for a replacement document version."""
    from src.storage.document_storage import normalise_storage_filename

    source_document_id = previous_document.get(
        "source_document_id",
        previous_document["document_id"],
    )
    safe_uploaded_filename = normalise_storage_filename(original_filename)

    return f"{source_document_id}_v{next_version_number}_{safe_uploaded_filename}"


@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """Answer a user question by calling the shared RAG engine."""
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if not is_meaningful_question(question):
        raise HTTPException(
            status_code=400,
            detail="Question must contain meaningful words, not only punctuation or symbols.",
        )

    try:
        from src.rag.engine import generate_answer

        result = generate_answer(
            question=question,
            role=request.role,
            department=request.department,
            department_filter=request.department_filter,
            file_type_filter=request.file_type_filter,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"RAG backend is unavailable: {error}",
        ) from error

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        role=request.role,
        department=request.department,
    )


@app.get("/admin/settings", response_model=SettingsResponse)
def get_admin_settings() -> SettingsResponse:
    """Return current backend-owned runtime settings for the admin UI."""
    current_settings = get_config_as_settings_dict(read_app_config())
    pending_settings = load_pending_runtime_settings()

    return SettingsResponse(
        settings=current_settings,
        pending_settings=pending_settings,
        rebuild_required=bool(pending_settings),
        changed_keys=list(pending_settings.keys()),
        message=(
            "Runtime settings loaded. Full rebuild is required for pending settings."
            if pending_settings
            else "Runtime settings loaded."
        ),
    )


@app.post("/admin/settings", response_model=SettingsUpdateResponse)
def update_admin_settings(request: SettingsUpdateRequest) -> SettingsUpdateResponse:
    """Validate and save backend-owned runtime settings."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can update runtime settings.",
        )

    current_settings = get_config_as_settings_dict(read_app_config())

    try:
        validate_runtime_settings(request.settings)

        changed_keys = find_changed_settings(
            current_settings,
            request.settings,
        )

        risky_settings = {
            key: value
            for key, value in request.settings.items()
            if settings_require_rebuild([key])
            and key in changed_keys
        }

        safe_settings = {
            key: value
            for key, value in request.settings.items()
            if key not in risky_settings
        }

        if safe_settings:
            save_runtime_settings(
                settings=safe_settings,
                updated_by=request.updated_by,
            )

        if risky_settings:
            save_pending_runtime_settings(
                settings=risky_settings,
                requested_by=request.updated_by,
            )

        updated_settings = get_config_as_settings_dict(read_app_config())
        pending_settings = load_pending_runtime_settings()
        rebuild_required = bool(pending_settings)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    if rebuild_required:
        message = (
            "Settings saved. Full search index rebuild is required before "
            "pending vector or embedding changes become active."
        )
    elif changed_keys:
        message = "Settings saved. No search index rebuild is required."
    else:
        message = "No setting changes detected."

    return SettingsUpdateResponse(
        status="saved",
        settings=updated_settings,
        pending_settings=pending_settings,
        changed_keys=changed_keys,
        rebuild_required=rebuild_required,
        message=message,
    )


def run_full_reindex(updated_by: str) -> dict:
    """Rebuild the target search index and promote pending config only after success."""
    import gc

    from src.evaluation.index_benchmark import (
        build_full_rebuild_benchmark,
        save_benchmark_result,
    )
    from src.vector.factory import (
        get_vector_backend,
        get_vector_backend_for_config,
    )

    active_backend = get_vector_backend()
    target_config = read_app_config_with_pending()
    target_backend = get_vector_backend_for_config(target_config)

    active_backend.clear_vector_store_cache()
    target_backend.clear_vector_store_cache()
    gc.collect()

    benchmark_result = build_full_rebuild_benchmark(config=target_config)
    save_benchmark_result(benchmark_result)
    result = benchmark_result["rebuild_result"]

    promoted_settings = promote_pending_runtime_settings(updated_by=updated_by)

    active_backend.clear_vector_store_cache()
    target_backend.clear_vector_store_cache()
    gc.collect()

    promotion_message = (
        f" Promoted pending setting(s): {', '.join(promoted_settings.keys())}."
        if promoted_settings
        else ""
    )

    return {
        "status": "success",
        "documents_indexed": result["documents_indexed"],
        "document_objects_loaded": result["document_objects_loaded"],
        "chunks_indexed": result["chunks_indexed"],
        "message": (
            f"Rebuilt search index with {result['documents_indexed']} file(s), "
            f"{result['document_objects_loaded']} document object(s), "
            f"and {result['chunks_indexed']} chunk(s)."
            f"{promotion_message}"
        ),
    }


@app.post("/admin/reindex", response_model=ReindexResponse)
def reindex_knowledge_base(request: ReindexRequest) -> ReindexResponse:
    """Synchronously rebuild the target search index for compatibility/testing."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can rebuild the vector index.",
        )

    try:
        result = run_full_reindex(updated_by=request.role)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Index rebuild failed. Active settings were not changed: {error}",
        ) from error

    return ReindexResponse(**result)


def run_pending_index_update(updated_by: str) -> dict:
    """Index active pending documents and persist benchmark/update metadata."""
    import time
    from pathlib import Path

    from src.etl.pipeline import index_changed_documents_with_cleanup
    from src.evaluation.index_benchmark import (
        build_index_benchmark_snapshot,
        calculate_index_delta,
        save_benchmark_result,
    )
    from src.metadata.repository import (
        load_pending_index_documents,
        load_replaced_documents_for_new_versions,
        mark_documents_indexed,
    )

    pending_documents = load_pending_index_documents()

    if not pending_documents:
        return {
            "status": "no_pending_documents",
            "pending_document_count": 0,
            "updated_sources": [],
            "total_deleted_vectors": 0,
            "total_chunks_indexed": 0,
            "elapsed_seconds": 0,
            "message": "No pending document updates require indexing.",
        }

    pending_document_ids = [
        document["document_id"]
        for document in pending_documents
    ]

    replaced_documents = load_replaced_documents_for_new_versions(
        pending_document_ids
    )

    index_source_paths = [
        str(Path("data/simulated") / document["filename"])
        for document in pending_documents
    ]

    replaced_source_paths = [
        str(Path("data/simulated") / document["filename"])
        for document in replaced_documents
    ]

    cleanup_source_paths = replaced_source_paths + index_source_paths

    before_snapshot = build_index_benchmark_snapshot()

    start_time = time.perf_counter()
    update_result = index_changed_documents_with_cleanup(
        index_source_paths=index_source_paths,
        cleanup_source_paths=cleanup_source_paths,
    )
    elapsed_seconds = round(time.perf_counter() - start_time, 3)

    after_snapshot = build_index_benchmark_snapshot()

    benchmark_result = {
        "benchmark_type": "batch_incremental_update",
        "changed_document_count": update_result["changed_document_count"],
        "updated_sources": update_result["updated_sources"],
        "cleanup_sources": update_result["cleanup_sources"],
        "elapsed_seconds": elapsed_seconds,
        "before": before_snapshot,
        "update_results": update_result["update_results"],
        "cleanup_results": update_result["cleanup_results"],
        "total_deleted_vectors": update_result["total_deleted_vectors"],
        "total_document_objects_loaded": update_result["total_document_objects_loaded"],
        "total_chunks_indexed": update_result["total_chunks_indexed"],
        "estimated_unchanged_chunks_avoided": max(
            before_snapshot["indexed_chunk_count"] - update_result["total_chunks_indexed"],
            0,
        ),
        "after": after_snapshot,
        "delta": calculate_index_delta(after_snapshot, before_snapshot),
        "updated_by": updated_by,
    }

    save_benchmark_result(benchmark_result)
    mark_documents_indexed(pending_document_ids)

    return {
        "status": "success",
        "pending_document_count": len(pending_documents),
        "updated_sources": update_result["updated_sources"],
        "total_deleted_vectors": update_result["total_deleted_vectors"],
        "total_chunks_indexed": update_result["total_chunks_indexed"],
        "elapsed_seconds": elapsed_seconds,
        "message": (
            f"Indexed {len(pending_documents)} pending document(s), refreshed "
            f"{update_result['total_chunks_indexed']} chunk(s), and replaced "
            f"{update_result['total_deleted_vectors']} old vector(s)."
        ),
    }


@app.post("/admin/index-updates", response_model=IndexUpdatesResponse)
def index_pending_document_updates(request: IndexUpdatesRequest) -> IndexUpdatesResponse:
    """Run incremental indexing for active documents marked as pending index."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can index pending document updates.",
        )

    try:
        result = run_pending_index_update(updated_by=request.role)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Pending index update failed: {error}",
        ) from error

    return IndexUpdatesResponse(**result)


@app.post("/admin/validate-upload", response_model=UploadValidationResponse)
def validate_upload_metadata(
    request: UploadValidationRequest,
) -> UploadValidationResponse:
    """Validate upload metadata permissions before local file/metadata writes."""
    return approve_upload_scope(
        role=request.role,
        user_department=request.user_department,
        document_department=request.document_department,
        allowed_roles=request.allowed_roles,
        allowed_departments=request.allowed_departments,
    )


@app.post("/admin/upload-document", response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    role: str = Form(...),
    user: str = Form(...),
    user_department: str = Form(...),
    title: str = Form(...),
    document_department: str = Form(...),
    category: str = Form("General"),
    tags_json: str = Form("[]"),
    allowed_roles_json: str = Form("[]"),
    allowed_departments_json: str = Form("[]"),
) -> UploadDocumentResponse:
    """Create one uploaded document through the backend-owned write path."""
    from src.metadata.repository import (
        append_document_metadata,
        generate_document_id,
        load_document_metadata,
        metadata_exists_for_filename,
    )
    from src.storage.document_storage import (
        normalise_storage_filename,
        save_document_bytes,
    )

    if not title.strip():
        raise HTTPException(
            status_code=400,
            detail="Document title is required.",
        )

    original_filename = file.filename or "uploaded_document"
    safe_filename = normalise_storage_filename(original_filename)
    file_type = get_uploaded_file_type(safe_filename)

    if metadata_exists_for_filename(safe_filename):
        raise HTTPException(
            status_code=409,
            detail="Metadata already exists for this filename.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file cannot be empty.",
        )

    allowed_roles = parse_json_list_field(allowed_roles_json, "allowed_roles_json")
    allowed_departments = parse_json_list_field(
        allowed_departments_json,
        "allowed_departments_json",
    )
    tags = parse_json_list_field(tags_json, "tags_json")

    approved_metadata = approve_upload_scope(
        role=role,
        user_department=user_department,
        document_department=document_department,
        allowed_roles=allowed_roles,
        allowed_departments=allowed_departments,
    )

    stored_document = save_document_bytes(safe_filename, file_bytes)
    existing_documents = load_document_metadata(include_inactive=True)
    document_id = generate_document_id(existing_documents)

    new_document = {
        "document_id": document_id,
        "title": title.strip(),
        "filename": stored_document.filename,
        "storage_backend": stored_document.storage_backend,
        "storage_uri": stored_document.storage_uri,
        "file_type": file_type,
        "source": "Manual Upload",
        "department": approved_metadata.document_department,
        "category": category.strip() or "General",
        "tags": tags,
        "allowed_roles": approved_metadata.allowed_roles,
        "allowed_departments": approved_metadata.allowed_departments,
        "uploaded_by": user,
        "uploaded_at": datetime.now().isoformat(timespec="minutes"),
        "page_number": None,
        "chunk_id": "pending_index",
        "visual_extraction_status": get_visual_extraction_status(file_type, file_bytes),
        "source_document_id": document_id,
        "version_number": 1,
        "is_active": 1,
        "content_hash": hashlib.sha256(file_bytes).hexdigest(),
        "archived_at": None,
        "replaced_by_document_id": None,
    }

    append_document_metadata(new_document)

    return UploadDocumentResponse(
        status="success",
        document_id=document_id,
        filename=stored_document.filename,
        storage_backend=stored_document.storage_backend,
        storage_uri=stored_document.storage_uri,
        chunk_id="pending_index",
        message=(
            f"Uploaded {stored_document.filename}, saved metadata record "
            f"{document_id}, and marked it pending index."
        ),
    )


@app.post("/admin/upload-zip-staging", response_model=BatchZipStageResponse)
async def upload_zip_for_staging(
    file: UploadFile = File(...),
    role: str = Form(...),
    user: str = Form(...),
    user_department: str = Form(...),
) -> BatchZipStageResponse:
    """Extract supported files from a ZIP and stage each for metadata review."""
    from src.connectors.graph_connector import (
        build_graph_document_id,
        build_graph_storage_filename,
        stage_graph_file_for_review,
    )
    from src.metadata.repository import load_document_metadata

    if role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot upload ZIP batches.",
        )

    original_zip_name = file.filename or "batch_upload.zip"

    if not original_zip_name.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Only ZIP files are supported for batch staging.",
        )

    zip_bytes = await file.read()

    if not zip_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded ZIP cannot be empty.",
        )

    existing_documents = load_document_metadata(include_inactive=True)
    existing_filenames = {
        document["filename"]
        for document in existing_documents
    }

    results = []

    try:
        zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid ZIP archive.",
        ) from error

    for zip_info in zip_file.infolist():
        if zip_info.is_dir():
            continue

        inner_path = zip_info.filename

        if inner_path.startswith("__MACOSX/") or Path(inner_path).name.startswith("."):
            continue

        inner_filename = Path(inner_path).name

        try:
            file_type = get_uploaded_file_type(inner_filename)
        except HTTPException:
            results.append(
                {
                    "File": inner_path,
                    "Status": "Skipped",
                    "Message": "Unsupported file type.",
                }
            )
            continue

        content_bytes = zip_file.read(zip_info)

        if not content_bytes:
            results.append(
                {
                    "File": inner_path,
                    "Status": "Skipped",
                    "Message": "Empty file.",
                }
            )
            continue

        visual_status = get_visual_extraction_status(file_type, content_bytes)
        source_path = f"zip://{original_zip_name}/{inner_path}"
        document_id = build_graph_document_id("ZIP", source_path)
        stored_filename = build_graph_storage_filename(
            source_type="zip",
            item_id=source_path,
            original_filename=inner_filename,
        )

        if stored_filename in existing_filenames:
            results.append(
                {
                    "File": inner_path,
                    "Status": "Skipped",
                    "Message": "A metadata record already exists for this stored filename.",
                }
            )
            continue

        try:
            metadata = stage_graph_file_for_review(
                document_id=document_id,
                title=Path(inner_filename).stem.replace("_", " "),
                original_filename=stored_filename,
                content_bytes=content_bytes,
                source_path=source_path,
                source_type="batch_zip",
                uploaded_by=user,
                visual_extraction_status=visual_status,
            )
        except Exception as error:
            results.append(
                {
                    "File": inner_path,
                    "Status": "Error",
                    "Message": str(error),
                }
            )
        else:
            existing_filenames.add(metadata["filename"])
            results.append(
                {
                    "File": inner_path,
                    "Status": "Staged",
                    "Document ID": metadata["document_id"],
                    "Department": metadata["department"],
                    "Visuals": metadata["visual_extraction_status"],
                    "Message": "Staged for metadata review.",
                }
            )

    staged_count = sum(1 for result in results if result["Status"] == "Staged")
    skipped_count = len(results) - staged_count

    return BatchZipStageResponse(
        status="success",
        staged_count=staged_count,
        skipped_count=skipped_count,
        results=results,
        message=f"Staged {staged_count} file(s) from ZIP; {skipped_count} skipped or failed.",
    )


@app.post("/admin/upload-document-version", response_model=UploadDocumentVersionResponse)
async def upload_document_version(
    file: UploadFile = File(...),
    role: str = Form(...),
    user: str = Form(...),
    user_department: str = Form(...),
    previous_document_id: str = Form(...),
) -> UploadDocumentVersionResponse:
    """Create a replacement version through the backend-owned write path."""
    from src.metadata.repository import (
        create_new_document_version,
        load_document_metadata,
        metadata_exists_for_filename,
    )
    from src.storage.document_storage import save_document_bytes

    if role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot upload document versions.",
        )

    all_documents = load_document_metadata(include_inactive=True)
    previous_document = next(
        (
            document for document in all_documents
            if document["document_id"] == previous_document_id
        ),
        None,
    )

    if previous_document is None:
        raise HTTPException(
            status_code=404,
            detail="Previous document was not found.",
        )

    if previous_document.get("is_active") == 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot replace an archived document version.",
        )

    if role == PROJECT_MANAGER_ROLE and previous_document["department"] != user_department:
        raise HTTPException(
            status_code=403,
            detail="Project Manager can only replace own-department documents.",
        )

    original_filename = file.filename or "replacement_document"
    previous_version_number = previous_document.get("version_number") or 1
    next_version_number = previous_version_number + 1
    stored_filename = build_versioned_filename(
        previous_document,
        original_filename,
        next_version_number,
    )
    file_type = get_uploaded_file_type(stored_filename)

    if metadata_exists_for_filename(stored_filename):
        raise HTTPException(
            status_code=409,
            detail="Metadata already exists for this replacement filename.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded replacement file cannot be empty.",
        )

    stored_document = save_document_bytes(stored_filename, file_bytes)
    new_document_id = generate_version_document_id(
        previous_document,
        next_version_number,
    )

    new_version_document = previous_document.copy()
    new_version_document.update(
        {
            "document_id": new_document_id,
            "filename": stored_document.filename,
            "storage_backend": stored_document.storage_backend,
            "storage_uri": stored_document.storage_uri,
            "file_type": file_type,
            "uploaded_by": user,
            "uploaded_at": datetime.now().isoformat(timespec="minutes"),
            "page_number": None,
            "chunk_id": "pending_index",
            "visual_extraction_status": get_visual_extraction_status(file_type, file_bytes),
            "content_hash": hashlib.sha256(file_bytes).hexdigest(),
        }
    )

    create_new_document_version(
        previous_document_id=previous_document["document_id"],
        new_document=new_version_document,
        archived_at=datetime.now().isoformat(timespec="minutes"),
    )

    return UploadDocumentVersionResponse(
        status="success",
        document_id=new_document_id,
        previous_document_id=previous_document["document_id"],
        filename=stored_document.filename,
        storage_backend=stored_document.storage_backend,
        storage_uri=stored_document.storage_uri,
        chunk_id="pending_index",
        version_number=next_version_number,
        message=(
            f"Created {previous_document['title']} v{next_version_number}, "
            "archived the previous version, and marked the new version pending index."
        ),
    )


def run_archive_document_action(request: ArchiveDocumentRequest) -> dict:
    """Archive one document and remove its vectors from the active index."""
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot archive knowledge base documents.",
        )

    try:
        from datetime import datetime
        from pathlib import Path

        from src.etl.pipeline import delete_vectors_for_source
        from src.metadata.repository import (
            archive_document_version,
            load_document_metadata,
        )

        all_documents = load_document_metadata(include_inactive=True)

        target_document = next(
            (
                document
                for document in all_documents
                if document["document_id"] == request.document_id
            ),
            None,
        )

        if target_document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        if target_document.get("is_active") == 0:
            raise HTTPException(
                status_code=400,
                detail="Document is already archived.",
            )

        if (
            request.role == PROJECT_MANAGER_ROLE
            and target_document["department"] != request.user_department
        ):
            raise HTTPException(
                status_code=403,
                detail="Project Manager can only archive own-department documents.",
            )

        source_path = str(Path("data/simulated") / target_document["filename"])

        archive_document_version(
            document_id=target_document["document_id"],
            replaced_by_document_id=None,
            archived_at=datetime.now().isoformat(timespec="minutes"),
        )

        deleted_vector_count = delete_vectors_for_source(source_path)

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document archive failed: {error}",
        ) from error

    return {
        "status": "success",
        "document_id": request.document_id,
        "deleted_vector_count": deleted_vector_count,
        "message": (
            f"Archived {target_document['title']} and removed "
            f"{deleted_vector_count} vector/index record(s) from the configured backend."
        ),
    }


def run_unarchive_document_action(request: UnarchiveDocumentRequest) -> dict:
    """Restore a manually archived document and mark it pending index."""
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot restore archived documents.",
        )

    try:
        from src.metadata.repository import (
            load_document_metadata,
            unarchive_document_version,
        )

        all_documents = load_document_metadata(include_inactive=True)

        target_document = next(
            (
                document
                for document in all_documents
                if document["document_id"] == request.document_id
            ),
            None,
        )

        if target_document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        if target_document.get("is_active") != 0:
            raise HTTPException(
                status_code=400,
                detail="Only archived documents can be restored.",
            )

        if target_document.get("replaced_by_document_id"):
            raise HTTPException(
                status_code=400,
                detail="Replaced old versions cannot be restored because a newer version exists.",
            )

        if request.role == PROJECT_MANAGER_ROLE:
            if target_document["department"] != request.user_department:
                raise HTTPException(
                    status_code=403,
                    detail="Project Manager can only restore own-department documents.",
                )

        unarchive_document_version(request.document_id)

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document restore failed: {error}",
        ) from error

    return {
        "status": "restored",
        "document_id": request.document_id,
        "chunk_id": "pending_index",
        "message": "Document restored and marked pending index.",
    }


def run_archive_document_job(job_id: str, request: ArchiveDocumentJobRequest) -> None:
    """Run document archive as a backend job and persist the outcome."""
    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        "Archiving document and removing search index records.",
    )

    try:
        result = run_archive_document_action(request)
    except HTTPException as error:
        update_job(
            job_id,
            JOB_STATUS_FAILED,
            f"Document archive failed: {error.detail}",
            {
                "status": "failed",
                "document_id": request.document_id,
                "message": str(error.detail),
            },
        )
    except Exception as error:
        update_job(
            job_id,
            JOB_STATUS_FAILED,
            f"Document archive failed: {error}",
            {
                "status": "failed",
                "document_id": request.document_id,
                "message": str(error),
            },
        )
    else:
        update_job(
            job_id,
            JOB_STATUS_SUCCEEDED,
            result["message"],
            result,
        )


def run_unarchive_document_job(job_id: str, request: UnarchiveDocumentJobRequest) -> None:
    """Run document restore as a backend job and persist the outcome."""
    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        "Restoring document and marking it pending index.",
    )

    try:
        result = run_unarchive_document_action(request)
    except HTTPException as error:
        update_job(
            job_id,
            JOB_STATUS_FAILED,
            f"Document restore failed: {error.detail}",
            {
                "status": "failed",
                "document_id": request.document_id,
                "message": str(error.detail),
            },
        )
    except Exception as error:
        update_job(
            job_id,
            JOB_STATUS_FAILED,
            f"Document restore failed: {error}",
            {
                "status": "failed",
                "document_id": request.document_id,
                "message": str(error),
            },
        )
    else:
        update_job(
            job_id,
            JOB_STATUS_SUCCEEDED,
            result["message"],
            result,
        )


@app.post("/admin/archive-document", response_model=ArchiveDocumentResponse)
def archive_document(request: ArchiveDocumentRequest) -> ArchiveDocumentResponse:
    """Synchronously archive one document for compatibility/testing."""
    return ArchiveDocumentResponse(**run_archive_document_action(request))


@app.post("/admin/unarchive-document", response_model=UnarchiveDocumentResponse)
def unarchive_document(
    request: UnarchiveDocumentRequest,
) -> UnarchiveDocumentResponse:
    """Synchronously restore one manually archived document for compatibility/testing."""
    return UnarchiveDocumentResponse(**run_unarchive_document_action(request))


@app.post("/admin/validate-metadata-update", response_model=MetadataUpdateValidationResponse)
def validate_metadata_update(
    request: MetadataUpdateValidationRequest,
) -> MetadataUpdateValidationResponse:
    """Validate metadata edit permissions before local SQLite metadata updates."""
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot edit knowledge base metadata.",
        )

    if request.role == PROJECT_MANAGER_ROLE:
        return MetadataUpdateValidationResponse(
            status="approved",
            document_department=request.user_department,
            allowed_roles=expand_allowed_roles(
                [
                    role for role in request.allowed_roles
                    if role in [PROJECT_MANAGER_ROLE, GENERAL_EMPLOYEE_ROLE]
                ] or [PROJECT_MANAGER_ROLE]
            ),
            allowed_departments=[request.user_department],
        )

    if request.role == SYSTEM_ADMIN_ROLE:
        if not request.allowed_roles:
            raise HTTPException(
                status_code=400,
                detail="At least one allowed role is required.",
            )

        if not request.allowed_departments:
            raise HTTPException(
                status_code=400,
                detail="At least one allowed department is required.",
            )

        return MetadataUpdateValidationResponse(
            status="approved",
            document_department=request.document_department,
            allowed_roles=expand_allowed_roles(request.allowed_roles),
            allowed_departments=expand_allowed_departments(request.allowed_departments),
        )

    raise HTTPException(
        status_code=403,
        detail="Unknown role cannot edit knowledge base metadata.",
    )


@app.post("/chat/jobs", response_model=JobResponse)
def create_chat_query_job(
    request: ChatJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a durable backend chat job so Streamlit reruns do not interrupt answers."""
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not is_meaningful_question(question):
        raise HTTPException(
            status_code=400,
            detail="Question must contain meaningful words, not only punctuation or symbols.",
        )

    job = create_job(
        job_type=JOB_TYPE_CHAT_QUERY,
        created_by=request.user,
        message="Chat query queued.",
    )

    background_tasks.add_task(run_chat_query_job, job["job_id"], request)

    return JobResponse(**job)


@app.get("/chat/sessions", response_model=list[ChatSessionResponse])
def get_chat_sessions(user: str) -> list[ChatSessionResponse]:
    """Return recent persisted chat sessions for one portal user."""
    return [
        ChatSessionResponse(**session)
        for session in list_chat_sessions_for_user(user)
    ]


@app.get("/chat/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
def get_chat_session_messages(
    session_id: str,
    user: str,
) -> list[ChatMessageResponse]:
    """Return persisted messages for a chat session owned by the requesting user."""
    return [
        ChatMessageResponse(**message)
        for message in list_chat_messages_for_session(session_id, user)
    ]


@app.get("/admin/jobs/latest", response_model=JobResponse | None)
def get_latest_backend_job(job_type: str | None = None) -> JobResponse | None:
    """Return the latest backend job, optionally filtered by type."""
    job = get_latest_job(job_type)

    if job is None:
        return None

    return JobResponse(**job)


@app.get("/admin/jobs/{job_id}", response_model=JobResponse)
def get_backend_job(job_id: str) -> JobResponse:
    """Return one backend job by ID."""
    job = get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return JobResponse(**job)


@app.post("/admin/reindex-jobs", response_model=JobResponse)
def create_reindex_job(
    request: ReindexJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a durable backend reindex job so Streamlit reruns do not interrupt rebuild."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can rebuild the vector index.",
        )

    job = create_job(
        job_type=JOB_TYPE_REINDEX,
        created_by=request.user,
        message="Search index rebuild queued.",
    )

    background_tasks.add_task(run_reindex_job, job["job_id"], request)

    return JobResponse(**job)


@app.post("/admin/index-update-jobs", response_model=JobResponse)
def create_index_update_job(
    request: IndexUpdatesJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a durable pending-index update job so Streamlit reruns do not interrupt indexing."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can index pending document updates.",
        )

    job = create_job(
        job_type=JOB_TYPE_INDEX_UPDATE,
        created_by=request.user,
        message="Pending document index update queued.",
    )

    background_tasks.add_task(run_index_update_job, job["job_id"], request)

    return JobResponse(**job)


@app.post("/admin/index-snapshot-jobs", response_model=JobResponse)
def create_index_snapshot_job(
    request: IndexSnapshotJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a durable job that refreshes the saved index snapshot."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can refresh the index snapshot.",
        )

    job = create_job(
        job_type=JOB_TYPE_INDEX_SNAPSHOT,
        created_by=request.user,
        message="Index snapshot refresh queued.",
    )

    background_tasks.add_task(run_index_snapshot_job, job["job_id"], request)

    return JobResponse(**job)


@app.post("/admin/archive-document-jobs", response_model=JobResponse)
def create_archive_document_job(
    request: ArchiveDocumentJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a durable archive job so Streamlit reruns do not interrupt cleanup."""
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot archive knowledge base documents.",
        )

    job = create_job(
        job_type=JOB_TYPE_DOCUMENT_ARCHIVE,
        created_by=request.user,
        message="Document archive queued.",
    )

    background_tasks.add_task(run_archive_document_job, job["job_id"], request)

    return JobResponse(**job)


@app.post("/admin/unarchive-document-jobs", response_model=JobResponse)
def create_unarchive_document_job(
    request: UnarchiveDocumentJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a durable restore job so Streamlit reruns do not interrupt updates."""
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot restore archived documents.",
        )

    job = create_job(
        job_type=JOB_TYPE_DOCUMENT_UNARCHIVE,
        created_by=request.user,
        message="Document restore queued.",
    )

    background_tasks.add_task(run_unarchive_document_job, job["job_id"], request)

    return JobResponse(**job)


@app.post("/admin/approve-document", response_model=ApproveDocumentResponse)
def approve_pending_document(
    request: ApproveDocumentRequest,
) -> ApproveDocumentResponse:
    """Approve a pending-review connector document and mark it for indexing."""
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot approve connector documents.",
        )

    try:
        from src.metadata.repository import (
            approve_document_for_indexing,
            load_document_metadata,
        )

        all_documents = load_document_metadata(include_inactive=True)

        target_document = next(
            (
                document
                for document in all_documents
                if document["document_id"] == request.document_id
            ),
            None,
        )

        if target_document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        if target_document.get("chunk_id") != "pending_review":
            raise HTTPException(
                status_code=400,
                detail="Only pending-review documents can be approved.",
            )

        if request.role == PROJECT_MANAGER_ROLE:
            if target_document["department"] != request.user_department:
                raise HTTPException(
                    status_code=403,
                    detail="Project Manager can only approve own-department documents.",
                )

        approved_metadata = approve_upload_scope(
            role=request.role,
            user_department=request.user_department,
            document_department=request.department,
            allowed_roles=request.allowed_roles,
            allowed_departments=request.allowed_departments,
        )

        updated_document = target_document.copy()
        visual_status = (
            target_document.get("visual_extraction_status")
            or get_visual_extraction_status(target_document["file_type"])
        )

        if str(target_document.get("source", "")).lower() == "onenote":
            visual_status = "Text extracted from OneNote page"

        updated_document.update(
            {
                "title": request.title.strip(),
                "department": approved_metadata.document_department,
                "category": request.category.strip() or "Connector Import",
                "tags": request.tags,
                "allowed_roles": approved_metadata.allowed_roles,
                "allowed_departments": approved_metadata.allowed_departments,
                "visual_extraction_status": visual_status,
            }
        )

        approve_document_for_indexing(
            document_id=request.document_id,
            updated_document=updated_document,
        )

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document approval failed: {error}",
        ) from error

    return ApproveDocumentResponse(
        status="approved",
        document_id=request.document_id,
        chunk_id="pending_index",
        message="Document approved and marked for search index update.",
    )


def graph_document_matches_source(document: dict, base_document_id: str) -> bool:
    """Match all metadata rows that belong to one external Graph source."""
    document_id = document.get("document_id") or ""
    source_document_id = document.get("source_document_id") or ""

    return (
        document_id == base_document_id
        or source_document_id == base_document_id
        or document_id.startswith(f"{base_document_id}-R")
        or document_id.startswith(f"{base_document_id}-V")
        or source_document_id.startswith(f"{base_document_id}-R")
        or source_document_id.startswith(f"{base_document_id}-V")
    )


def get_onenote_connector_state(page_id: str, documents: list[dict]) -> dict:
    """Return current review/index state for a discovered OneNote page."""
    from src.connectors.graph_connector import build_graph_document_id

    base_document_id = build_graph_document_id("GRAPH-ON", page_id)

    matching_documents = [
        document
        for document in documents
        if graph_document_matches_source(document, base_document_id)
    ]

    if not matching_documents:
        return {
            "connector_state": "New",
            "staged_document_id": None,
        }

    active_matches = [
        document
        for document in matching_documents
        if document.get("is_active") == 1
    ]

    latest_document = sorted(
        active_matches or matching_documents,
        key=lambda document: document.get("version_number") or 1,
    )[-1]

    chunk_id = latest_document.get("chunk_id")

    state_by_chunk_id = {
        "pending_review": "Pending Review",
        "rejected": "Rejected",
        "pending_index": "Pending Index",
        "pending": "Pending Index",
        "indexed": "Indexed",
        "archived": "Archived",
    }

    return {
        "connector_state": state_by_chunk_id.get(chunk_id, "Other"),
        "staged_document_id": latest_document["document_id"],
    }


def find_active_graph_document(
    documents: list[dict],
    base_document_id: str,
) -> dict | None:
    """Find the active KB document for a Graph source identity."""
    matching_documents = [
        document
        for document in documents
        if document.get("is_active") == 1
        and graph_document_matches_source(document, base_document_id)
    ]

    if not matching_documents:
        return None

    return sorted(
        matching_documents,
        key=lambda document: document.get("version_number") or 1,
    )[-1]


def get_onedrive_connector_state(item_id: str, documents: list[dict]) -> dict:
    """Return current review/index state for a discovered OneDrive file."""
    from src.connectors.graph_connector import build_graph_document_id

    base_document_id = build_graph_document_id("GRAPH-OD", item_id)

    matching_documents = [
        document
        for document in documents
        if graph_document_matches_source(document, base_document_id)
    ]

    if not matching_documents:
        return {
            "connector_state": "New",
            "staged_document_id": None,
        }

    active_matches = [
        document
        for document in matching_documents
        if document.get("is_active") == 1
    ]

    latest_document = sorted(
        active_matches or matching_documents,
        key=lambda document: document.get("version_number") or 1,
    )[-1]

    chunk_id = latest_document.get("chunk_id")

    state_by_chunk_id = {
        "pending_review": "Pending Review",
        "rejected": "Rejected",
        "pending_index": "Pending Index",
        "pending": "Pending Index",
        "indexed": "Indexed",
        "archived": "Archived",
    }

    return {
        "connector_state": state_by_chunk_id.get(chunk_id, "Other"),
        "staged_document_id": latest_document["document_id"],
    }


@app.post("/admin/graph/onedrive/files", response_model=OneDriveFilesResponse)
def list_graph_onedrive_files(
    request: GraphConnectorListRequest,
) -> OneDriveFilesResponse:
    """List files under the configured OneDrive connector root."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can scan OneDrive connector files.",
        )

    try:
        from src.connectors.graph_client import list_onedrive_files_recursive
        from src.metadata.repository import load_document_metadata

        discovered_files = list_onedrive_files_recursive()
        all_documents = load_document_metadata(include_inactive=True)

    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"OneDrive scan failed: {error}",
        ) from error

    files = []

    for file_item in discovered_files:
        connector_state = get_onedrive_connector_state(
            item_id=file_item["id"],
            documents=all_documents,
        )

        files.append(
            OneDriveFileSummary(
                id=file_item["id"],
                name=file_item["name"],
                connector_path=file_item["connector_path"],
                size=file_item.get("size"),
                last_modified_datetime=file_item.get("lastModifiedDateTime"),
                connector_state=connector_state["connector_state"],
                staged_document_id=connector_state["staged_document_id"],
            )
        )

    return OneDriveFilesResponse(status="success", files=files)


@app.post("/admin/graph/onenote/pages", response_model=OneNotePagesResponse)
def list_graph_onenote_pages(
    request: GraphConnectorListRequest,
) -> OneNotePagesResponse:
    """List OneNote pages available to the signed-in Graph user."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can scan OneNote connector pages.",
        )

    try:
        from src.connectors.graph_client import list_onenote_pages_recursive
        from src.metadata.repository import load_document_metadata

        discovered_pages = list_onenote_pages_recursive()
        all_documents = load_document_metadata(include_inactive=True)

    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"OneNote scan failed: {error}",
        ) from error

    pages = []

    for page in discovered_pages:
        connector_state = get_onenote_connector_state(
            page_id=page["id"],
            documents=all_documents,
        )

        pages.append(
            OneNotePageSummary(
                id=page["id"],
                title=page.get("title", "Untitled Page"),
                notebook_name=page.get("notebook_name", "Untitled Notebook"),
                section_name=page.get("section_name", "Untitled Section"),
                connector_path=page["connector_path"],
                last_modified_datetime=page.get("lastModifiedDateTime"),
                connector_state=connector_state["connector_state"],
                staged_document_id=connector_state["staged_document_id"],
            )
        )

    return OneNotePagesResponse(status="success", pages=pages)


@app.post("/admin/graph/onedrive/stage-file", response_model=StageOneDriveFileResponse)
def stage_onedrive_file(
    request: StageOneDriveFileRequest,
) -> StageOneDriveFileResponse:
    """Download one OneDrive file and stage it for admin metadata review."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can stage OneDrive connector files.",
        )

    try:
        from src.connectors.graph_client import download_onedrive_file_by_item_id
        from src.connectors.graph_connector import (
            build_graph_document_id,
            build_graph_storage_filename,
            stage_graph_file_for_review,
        )
        from src.metadata.repository import load_document_metadata

        document_id = build_graph_document_id("GRAPH-OD", request.item_id)

        existing_documents = load_document_metadata(include_inactive=True)
        base_document_id = build_graph_document_id("GRAPH-OD", request.item_id)

        matching_documents = [
            document
            for document in existing_documents
            if (
                document["document_id"] == base_document_id
                or document["document_id"].startswith(f"{base_document_id}-R")
            )
        ]

        blocking_document = next(
            (
                document
                for document in matching_documents
                if document.get("chunk_id") != "rejected"
            ),
            None,
        )

        if blocking_document:
            raise HTTPException(
                status_code=409,
                detail="This OneDrive file has already been staged or approved.",
            )

        document_id = base_document_id
        attempt_suffix = ""

        if matching_documents:
            next_attempt_number = len(matching_documents) + 1
            document_id = f"{base_document_id}-R{next_attempt_number}"
            attempt_suffix = f"_r{next_attempt_number}"

        content_bytes = download_onedrive_file_by_item_id(request.item_id)

        if not content_bytes:
            raise HTTPException(
                status_code=400,
                detail="Downloaded OneDrive file is empty.",
            )

        stored_filename = build_graph_storage_filename(
            source_type="onedrive",
            item_id=f"{request.item_id}{attempt_suffix}",
            original_filename=request.name,
        )
        file_type = get_uploaded_file_type(stored_filename)
        visual_status = get_visual_extraction_status(file_type, content_bytes)

        metadata = stage_graph_file_for_review(
            document_id=document_id,
            title=request.name,
            original_filename=stored_filename,
            content_bytes=content_bytes,
            source_path=request.connector_path,
            source_type="onedrive",
            uploaded_by=request.user,
            source_document_id=base_document_id,
            visual_extraction_status=visual_status,
        )

    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"OneDrive staging failed: {error}",
        ) from error

    return StageOneDriveFileResponse(
        status="staged",
        document_id=metadata["document_id"],
        filename=metadata["filename"],
        storage_backend=metadata["storage_backend"],
        storage_uri=metadata["storage_uri"],
        chunk_id=metadata["chunk_id"],
        message="OneDrive file downloaded and staged for metadata review.",
    )


@app.post("/admin/reject-staged-document")
def reject_staged_document(request: RejectStagedDocumentRequest) -> dict:
    """Reject a staged connector document before it enters active indexing."""
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot reject connector documents.",
        )

    try:
        from src.metadata.repository import (
            load_document_metadata,
            reject_pending_review_document,
        )

        all_documents = load_document_metadata(include_inactive=True)

        target_document = next(
            (
                document
                for document in all_documents
                if document["document_id"] == request.document_id
            ),
            None,
        )

        if target_document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        if target_document.get("chunk_id") != "pending_review":
            raise HTTPException(
                status_code=400,
                detail="Only pending-review documents can be rejected.",
            )

        if request.role == PROJECT_MANAGER_ROLE:
            if target_document["department"] != request.user_department:
                raise HTTPException(
                    status_code=403,
                    detail="Project Manager can only reject own-department documents.",
                )

        reject_pending_review_document(
            document_id=request.document_id,
            rejected_at=datetime.now().isoformat(timespec="minutes"),
        )

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document rejection failed: {error}",
        ) from error

    return {
        "status": "rejected",
        "document_id": request.document_id,
        "message": "Staged connector document rejected and removed from pending review.",
    }


def stage_onedrive_file_item(
    item_id: str,
    name: str,
    connector_path: str,
    user: str,
) -> dict:
    """Download and stage one OneDrive file item for pending metadata review."""
    from src.connectors.graph_client import download_onedrive_file_by_item_id
    from src.connectors.graph_connector import (
        build_graph_document_id,
        build_graph_storage_filename,
        stage_graph_file_for_review,
    )
    from src.metadata.repository import load_document_metadata

    existing_documents = load_document_metadata(include_inactive=True)
    base_document_id = build_graph_document_id("GRAPH-OD", item_id)

    matching_documents = [
        document
        for document in existing_documents
        if (
            document["document_id"] == base_document_id
            or document["document_id"].startswith(f"{base_document_id}-R")
        )
    ]

    blocking_document = next(
        (
            document
            for document in matching_documents
            if document.get("chunk_id") != "rejected"
        ),
        None,
    )

    if blocking_document:
        raise ValueError("This OneDrive file has already been staged or approved.")

    document_id = base_document_id
    attempt_suffix = ""

    if matching_documents:
        next_attempt_number = len(matching_documents) + 1
        document_id = f"{base_document_id}-R{next_attempt_number}"
        attempt_suffix = f"_r{next_attempt_number}"

    content_bytes = download_onedrive_file_by_item_id(item_id)

    if not content_bytes:
        raise ValueError("Downloaded OneDrive file is empty.")

    stored_filename = build_graph_storage_filename(
        source_type="onedrive",
        item_id=f"{item_id}{attempt_suffix}",
        original_filename=name,
    )
    file_type = get_uploaded_file_type(stored_filename)
    visual_status = get_visual_extraction_status(file_type, content_bytes)

    return stage_graph_file_for_review(
        document_id=document_id,
        title=name,
        original_filename=stored_filename,
        content_bytes=content_bytes,
        source_path=connector_path,
        source_type="onedrive",
        uploaded_by=user,
        source_document_id=base_document_id,
        visual_extraction_status=visual_status,
    )


def run_onedrive_stage_job(job_id: str, request: StageOneDriveFilesJobRequest) -> None:
    """Stage selected OneDrive files in the background and store per-file results."""
    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        f"Staging {len(request.files)} OneDrive file(s).",
    )

    results = []

    for file_item in request.files:
        try:
            metadata = stage_onedrive_file_item(
                item_id=file_item.item_id,
                name=file_item.name,
                connector_path=file_item.connector_path,
                user=request.user,
            )
        except Exception as error:
            results.append(
                {
                    "File": file_item.name,
                    "Path": file_item.connector_path,
                    "Status": "Rejected",
                    "Message": str(error),
                }
            )
        else:
            results.append(
                {
                    "File": file_item.name,
                    "Path": file_item.connector_path,
                    "Status": "Staged",
                    "Message": f"Staged {metadata['document_id']} for review.",
                }
            )

    staged_count = sum(1 for result in results if result["Status"] == "Staged")

    update_job(
        job_id,
        JOB_STATUS_SUCCEEDED,
        f"OneDrive staging finished: {staged_count}/{len(results)} file(s) staged.",
        result={
            "results": results,
            "staged_count": staged_count,
            "total_count": len(results),
            "message": f"OneDrive staging finished: {staged_count}/{len(results)} file(s) staged.",
        },
    )


@app.post("/admin/graph/onedrive/stage-files-job", response_model=JobResponse)
def create_onedrive_stage_job(
    request: StageOneDriveFilesJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a durable job for batch OneDrive file staging."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can batch stage OneDrive connector files.",
        )

    if not request.files:
        raise HTTPException(
            status_code=400,
            detail="At least one OneDrive file is required.",
        )

    job = create_job(
        job_type=JOB_TYPE_ONEDRIVE_STAGE,
        created_by=request.user,
        message=f"OneDrive staging queued for {len(request.files)} file(s).",
    )

    background_tasks.add_task(run_onedrive_stage_job, job["job_id"], request)

    return JobResponse(**job)


def refresh_onedrive_file_item(
    item_id: str,
    name: str,
    connector_path: str,
    role: str,
    user: str,
    user_department: str,
) -> RefreshOneDriveFileResponse:
    """Refresh one OneDrive-backed KB document and return a structured result."""
    from src.connectors.graph_client import download_onedrive_file_by_item_id
    from src.connectors.graph_connector import (
        build_graph_document_id,
        build_graph_storage_filename,
    )
    from src.metadata.repository import (
        create_new_document_version,
        load_document_metadata,
    )
    from src.storage.document_storage import save_document_bytes

    base_document_id = build_graph_document_id("GRAPH-OD", item_id)
    all_documents = load_document_metadata(include_inactive=True)

    active_document = find_active_graph_document(
        documents=all_documents,
        base_document_id=base_document_id,
    )

    if active_document is None:
        raise HTTPException(
            status_code=404,
            detail="No active KB document found for this OneDrive source item.",
        )

    if active_document.get("source") != "onedrive":
        raise HTTPException(
            status_code=400,
            detail="Only OneDrive-backed documents can be refreshed by this endpoint.",
        )

    if active_document.get("chunk_id") not in ["pending_index", "indexed"]:
        raise HTTPException(
            status_code=400,
            detail="Only approved OneDrive documents can be refreshed.",
        )

    if role == PROJECT_MANAGER_ROLE and active_document["department"] != user_department:
        raise HTTPException(
            status_code=403,
            detail="Project Manager can only refresh own-department connector documents.",
        )

    content_bytes = download_onedrive_file_by_item_id(item_id)

    if not content_bytes:
        raise HTTPException(
            status_code=400,
            detail="Downloaded OneDrive file is empty.",
        )

    new_content_hash = hashlib.sha256(content_bytes).hexdigest()

    if active_document.get("content_hash") == new_content_hash:
        return RefreshOneDriveFileResponse(
            status="no_change",
            document_id=active_document["document_id"],
            chunk_id=active_document.get("chunk_id"),
            message="No content change detected for this OneDrive document.",
        )

    next_version_number = (active_document.get("version_number") or 1) + 1
    source_document_id = base_document_id
    new_document_id = f"{source_document_id}-V{next_version_number}"

    stored_filename = build_graph_storage_filename(
        source_type="onedrive",
        item_id=f"{item_id}_v{next_version_number}",
        original_filename=name,
    )

    stored_document = save_document_bytes(stored_filename, content_bytes)
    file_type = get_uploaded_file_type(stored_document.filename)

    new_document = active_document.copy()
    new_document.update(
        {
            "document_id": new_document_id,
            "title": name,
            "filename": stored_document.filename,
            "storage_backend": stored_document.storage_backend,
            "storage_uri": stored_document.storage_uri,
            "file_type": file_type,
            "source": "onedrive",
            "source_document_id": source_document_id,
            "uploaded_by": user,
            "uploaded_at": datetime.now().isoformat(timespec="minutes"),
            "chunk_id": "pending_index",
            "visual_extraction_status": get_visual_extraction_status(file_type, content_bytes),
            "content_hash": new_content_hash,
            "archived_at": None,
            "replaced_by_document_id": None,
        }
    )

    create_new_document_version(
        previous_document_id=active_document["document_id"],
        new_document=new_document,
        archived_at=datetime.now().isoformat(timespec="minutes"),
    )

    return RefreshOneDriveFileResponse(
        status="updated",
        document_id=new_document_id,
        previous_document_id=active_document["document_id"],
        chunk_id="pending_index",
        message=(
            f"Detected OneDrive content change and created {new_document_id}. "
            "Run Update for Pending Documents to refresh search results."
        ),
    )


@app.post("/admin/graph/onedrive/refresh-file", response_model=RefreshOneDriveFileResponse)
def refresh_onedrive_file(
    request: RefreshOneDriveFileRequest,
) -> RefreshOneDriveFileResponse:
    """Refresh one OneDrive-backed KB document if the Graph source changed."""
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot refresh connector documents.",
        )

    try:
        return refresh_onedrive_file_item(
            item_id=request.item_id,
            name=request.name,
            connector_path=request.connector_path,
            role=request.role,
            user=request.user,
            user_department=request.user_department,
        )
    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"OneDrive refresh failed: {error}",
        ) from error


def run_onedrive_refresh_job(
    job_id: str,
    request: RefreshOneDriveFilesJobRequest,
) -> None:
    """Refresh selected OneDrive files in the background and store per-file results."""
    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        f"Refreshing {len(request.files)} OneDrive file(s).",
    )

    results = []

    for file_item in request.files:
        try:
            refresh_result = refresh_onedrive_file_item(
                item_id=file_item.item_id,
                name=file_item.name,
                connector_path=file_item.connector_path,
                role=request.role,
                user=request.user,
                user_department=request.user_department,
            )
        except HTTPException as error:
            results.append(
                {
                    "File": file_item.name,
                    "Path": file_item.connector_path,
                    "Status": "Rejected",
                    "Message": str(error.detail),
                }
            )
        except Exception as error:
            results.append(
                {
                    "File": file_item.name,
                    "Path": file_item.connector_path,
                    "Status": "Error",
                    "Message": str(error),
                }
            )
        else:
            status_label = (
                "Updated"
                if refresh_result.status == "updated"
                else "No Change"
            )
            results.append(
                {
                    "File": file_item.name,
                    "Path": file_item.connector_path,
                    "Status": status_label,
                    "Message": refresh_result.message,
                }
            )

    updated_count = sum(1 for result in results if result["Status"] == "Updated")
    unchanged_count = sum(1 for result in results if result["Status"] == "No Change")

    update_job(
        job_id,
        JOB_STATUS_SUCCEEDED,
        (
            f"OneDrive refresh finished: {updated_count} updated, "
            f"{unchanged_count} unchanged, {len(results)} checked."
        ),
        result={
            "results": results,
            "updated_count": updated_count,
            "unchanged_count": unchanged_count,
            "total_count": len(results),
            "message": (
                f"OneDrive refresh finished: {updated_count} updated, "
                f"{unchanged_count} unchanged, {len(results)} checked."
            ),
        },
    )


@app.post("/admin/graph/onedrive/refresh-files-job", response_model=JobResponse)
def create_onedrive_refresh_job(
    request: RefreshOneDriveFilesJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a durable job for batch OneDrive source refresh."""
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot refresh connector documents.",
        )

    if not request.files:
        raise HTTPException(
            status_code=400,
            detail="At least one OneDrive file is required.",
        )

    job = create_job(
        job_type=JOB_TYPE_ONEDRIVE_REFRESH,
        created_by=request.user,
        message=f"OneDrive refresh queued for {len(request.files)} file(s).",
    )

    background_tasks.add_task(run_onedrive_refresh_job, job["job_id"], request)

    return JobResponse(**job)


@app.post("/admin/graph/onenote/stage-page", response_model=StageOneNotePageResponse)
def stage_onenote_page(
    request: StageOneNotePageRequest,
) -> StageOneNotePageResponse:
    """Download one OneNote page and stage it for admin metadata review."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can stage OneNote connector pages.",
        )

    try:
        from src.connectors.graph_client import download_onenote_page_content_by_id
        from src.connectors.graph_connector import (
            build_graph_document_id,
            build_graph_storage_filename,
            normalize_onenote_html_to_text,
            stage_graph_file_for_review,
        )
        from src.metadata.repository import load_document_metadata

        existing_documents = load_document_metadata(include_inactive=True)
        document_id = build_graph_document_id("GRAPH-ON", request.page_id)

        if any(document["document_id"] == document_id for document in existing_documents):
            raise HTTPException(
                status_code=409,
                detail="This OneNote page has already been staged.",
            )

        html_content = download_onenote_page_content_by_id(request.page_id)
        text_content = normalize_onenote_html_to_text(html_content)

        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail="OneNote page did not contain extractable text.",
            )

        stored_filename = build_graph_storage_filename(
            source_type="onenote",
            item_id=request.page_id,
            original_filename=f"{request.title or 'untitled_onenote_page'}.txt",
        )

        metadata = stage_graph_file_for_review(
            document_id=document_id,
            title=request.title or "Untitled OneNote Page",
            original_filename=stored_filename,
            content_bytes=text_content.encode("utf-8"),
            source_path=request.connector_path,
            source_type="onenote",
            uploaded_by=request.user,
            source_document_id=document_id,
            visual_extraction_status="Text extracted from OneNote page",
        )

    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"OneNote staging failed: {error}",
        ) from error

    return StageOneNotePageResponse(
        status="staged",
        document_id=metadata["document_id"],
        filename=metadata["filename"],
        storage_backend=metadata["storage_backend"],
        storage_uri=metadata["storage_uri"],
        chunk_id=metadata["chunk_id"],
        message="OneNote page downloaded and staged for metadata review.",
    )


def stage_onenote_page_item(
    page_id: str,
    title: str,
    connector_path: str,
    user: str,
) -> dict:
    """Download and stage one OneNote page item for pending metadata review."""
    from src.connectors.graph_client import download_onenote_page_content_by_id
    from src.connectors.graph_connector import (
        build_graph_document_id,
        build_graph_storage_filename,
        normalize_onenote_html_to_text,
        stage_graph_file_for_review,
    )
    from src.metadata.repository import load_document_metadata

    existing_documents = load_document_metadata(include_inactive=True)
    base_document_id = build_graph_document_id("GRAPH-ON", page_id)

    matching_documents = [
        document
        for document in existing_documents
        if (
            document["document_id"] == base_document_id
            or document["document_id"].startswith(f"{base_document_id}-R")
        )
    ]

    blocking_document = next(
        (
            document
            for document in matching_documents
            if document.get("chunk_id") != "rejected"
        ),
        None,
    )

    if blocking_document:
        raise ValueError("This OneNote page has already been staged or approved.")

    document_id = base_document_id
    attempt_suffix = ""

    if matching_documents:
        next_attempt_number = len(matching_documents) + 1
        document_id = f"{base_document_id}-R{next_attempt_number}"
        attempt_suffix = f"_r{next_attempt_number}"

    html_content = download_onenote_page_content_by_id(page_id)
    text_content = normalize_onenote_html_to_text(html_content)

    if not text_content.strip():
        raise ValueError("OneNote page did not contain extractable text.")

    stored_filename = build_graph_storage_filename(
        source_type="onenote",
        item_id=f"{page_id}{attempt_suffix}",
        original_filename=f"{title or 'untitled_onenote_page'}.txt",
    )

    return stage_graph_file_for_review(
        document_id=document_id,
        title=title or "Untitled OneNote Page",
        original_filename=stored_filename,
        content_bytes=text_content.encode("utf-8"),
        source_path=connector_path,
        source_type="onenote",
        uploaded_by=user,
        source_document_id=base_document_id,
        visual_extraction_status="Text extracted from OneNote page",
    )


def run_onenote_stage_job(job_id: str, request: StageOneNotePagesJobRequest) -> None:
    """Stage selected OneNote pages in the background and store per-page results."""
    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        f"Staging {len(request.pages)} OneNote page(s).",
    )

    results = []

    for page_item in request.pages:
        try:
            metadata = stage_onenote_page_item(
                page_id=page_item.page_id,
                title=page_item.title,
                connector_path=page_item.connector_path,
                user=request.user,
            )
        except Exception as error:
            results.append(
                {
                    "Page": page_item.title or "Untitled Page",
                    "Path": page_item.connector_path,
                    "Status": "Rejected",
                    "Message": str(error),
                }
            )
        else:
            results.append(
                {
                    "Page": page_item.title or "Untitled Page",
                    "Path": page_item.connector_path,
                    "Status": "Staged",
                    "Message": f"Staged {metadata['document_id']} for review.",
                }
            )

    staged_count = sum(1 for result in results if result["Status"] == "Staged")

    update_job(
        job_id,
        JOB_STATUS_SUCCEEDED,
        f"OneNote staging finished: {staged_count}/{len(results)} page(s) staged.",
        result={
            "results": results,
            "staged_count": staged_count,
            "total_count": len(results),
            "message": f"OneNote staging finished: {staged_count}/{len(results)} page(s) staged.",
        },
    )


@app.post("/admin/graph/onenote/stage-pages-job", response_model=JobResponse)
def create_onenote_stage_job(
    request: StageOneNotePagesJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a durable job for batch OneNote page staging."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can batch stage OneNote connector pages.",
        )

    if not request.pages:
        raise HTTPException(
            status_code=400,
            detail="At least one OneNote page is required.",
        )

    job = create_job(
        job_type=JOB_TYPE_ONENOTE_STAGE,
        created_by=request.user,
        message=f"OneNote staging queued for {len(request.pages)} page(s).",
    )

    background_tasks.add_task(run_onenote_stage_job, job["job_id"], request)

    return JobResponse(**job)


def refresh_onenote_page_item(
    page_id: str,
    title: str,
    connector_path: str,
    role: str,
    user: str,
    user_department: str,
) -> RefreshOneNotePageResponse:
    """Refresh one OneNote-backed KB document and return a structured result."""
    from src.connectors.graph_client import download_onenote_page_content_by_id
    from src.connectors.graph_connector import (
        build_graph_document_id,
        build_graph_storage_filename,
        normalize_onenote_html_to_text,
    )
    from src.metadata.repository import (
        create_new_document_version,
        load_document_metadata,
    )
    from src.storage.document_storage import save_document_bytes

    base_document_id = build_graph_document_id("GRAPH-ON", page_id)
    all_documents = load_document_metadata(include_inactive=True)

    active_document = find_active_graph_document(
        documents=all_documents,
        base_document_id=base_document_id,
    )

    if active_document is None:
        raise HTTPException(
            status_code=404,
            detail="No active KB document found for this OneNote source page.",
        )

    if active_document.get("source") != "onenote":
        raise HTTPException(
            status_code=400,
            detail="Only OneNote-backed documents can be refreshed by this endpoint.",
        )

    if active_document.get("chunk_id") not in ["pending_index", "indexed"]:
        raise HTTPException(
            status_code=400,
            detail="Only approved OneNote documents can be refreshed.",
        )

    if role == PROJECT_MANAGER_ROLE and active_document["department"] != user_department:
        raise HTTPException(
            status_code=403,
            detail="Project Manager can only refresh own-department connector documents.",
        )

    html_content = download_onenote_page_content_by_id(page_id)
    text_content = normalize_onenote_html_to_text(html_content)

    if not text_content.strip():
        raise HTTPException(
            status_code=400,
            detail="OneNote page did not contain extractable text.",
        )

    content_bytes = text_content.encode("utf-8")
    new_content_hash = hashlib.sha256(content_bytes).hexdigest()

    if active_document.get("content_hash") == new_content_hash:
        return RefreshOneNotePageResponse(
            status="no_change",
            document_id=active_document["document_id"],
            chunk_id=active_document.get("chunk_id"),
            message="No content change detected for this OneNote page.",
        )

    next_version_number = (active_document.get("version_number") or 1) + 1
    source_document_id = base_document_id
    new_document_id = f"{source_document_id}-V{next_version_number}"

    stored_filename = build_graph_storage_filename(
        source_type="onenote",
        item_id=f"{page_id}_v{next_version_number}",
        original_filename=f"{title or 'untitled_onenote_page'}.txt",
    )

    stored_document = save_document_bytes(stored_filename, content_bytes)

    new_document = active_document.copy()
    new_document.update(
        {
            "document_id": new_document_id,
            "title": title or "Untitled OneNote Page",
            "filename": stored_document.filename,
            "storage_backend": stored_document.storage_backend,
            "storage_uri": stored_document.storage_uri,
            "file_type": "TXT",
            "source": "onenote",
            "source_document_id": source_document_id,
            "uploaded_by": user,
            "uploaded_at": datetime.now().isoformat(timespec="minutes"),
            "chunk_id": "pending_index",
            "visual_extraction_status": "Text extracted from OneNote page",
            "content_hash": new_content_hash,
            "archived_at": None,
            "replaced_by_document_id": None,
        }
    )

    create_new_document_version(
        previous_document_id=active_document["document_id"],
        new_document=new_document,
        archived_at=datetime.now().isoformat(timespec="minutes"),
    )

    return RefreshOneNotePageResponse(
        status="updated",
        document_id=new_document_id,
        previous_document_id=active_document["document_id"],
        chunk_id="pending_index",
        message=(
            f"Detected OneNote page change and created {new_document_id}. "
            "Run Update for Pending Documents to refresh search results."
        ),
    )


@app.post("/admin/graph/onenote/refresh-page", response_model=RefreshOneNotePageResponse)
def refresh_onenote_page(
    request: RefreshOneNotePageRequest,
) -> RefreshOneNotePageResponse:
    """Refresh one OneNote-backed KB document if the Graph source changed."""
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot refresh connector documents.",
        )

    try:
        return refresh_onenote_page_item(
            page_id=request.page_id,
            title=request.title,
            connector_path=request.connector_path,
            role=request.role,
            user=request.user,
            user_department=request.user_department,
        )
    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"OneNote refresh failed: {error}",
        ) from error


def run_onenote_refresh_job(
    job_id: str,
    request: RefreshOneNotePagesJobRequest,
) -> None:
    """Refresh selected OneNote pages in the background and store per-page results."""
    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        f"Refreshing {len(request.pages)} OneNote page(s).",
    )

    results = []

    for page_item in request.pages:
        try:
            refresh_result = refresh_onenote_page_item(
                page_id=page_item.page_id,
                title=page_item.title,
                connector_path=page_item.connector_path,
                role=request.role,
                user=request.user,
                user_department=request.user_department,
            )
        except HTTPException as error:
            results.append(
                {
                    "Page": page_item.title or "Untitled Page",
                    "Path": page_item.connector_path,
                    "Status": "Rejected",
                    "Message": str(error.detail),
                }
            )
        except Exception as error:
            results.append(
                {
                    "Page": page_item.title or "Untitled Page",
                    "Path": page_item.connector_path,
                    "Status": "Error",
                    "Message": str(error),
                }
            )
        else:
            status_label = (
                "Updated"
                if refresh_result.status == "updated"
                else "No Change"
            )
            results.append(
                {
                    "Page": page_item.title or "Untitled Page",
                    "Path": page_item.connector_path,
                    "Status": status_label,
                    "Message": refresh_result.message,
                }
            )

    updated_count = sum(1 for result in results if result["Status"] == "Updated")
    unchanged_count = sum(1 for result in results if result["Status"] == "No Change")

    update_job(
        job_id,
        JOB_STATUS_SUCCEEDED,
        (
            f"OneNote refresh finished: {updated_count} updated, "
            f"{unchanged_count} unchanged, {len(results)} checked."
        ),
        result={
            "results": results,
            "updated_count": updated_count,
            "unchanged_count": unchanged_count,
            "total_count": len(results),
            "message": (
                f"OneNote refresh finished: {updated_count} updated, "
                f"{unchanged_count} unchanged, {len(results)} checked."
            ),
        },
    )


@app.post("/admin/graph/onenote/refresh-pages-job", response_model=JobResponse)
def create_onenote_refresh_job(
    request: RefreshOneNotePagesJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a durable job for batch OneNote page refresh."""
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot refresh connector documents.",
        )

    if not request.pages:
        raise HTTPException(
            status_code=400,
            detail="At least one OneNote page is required.",
        )

    job = create_job(
        job_type=JOB_TYPE_ONENOTE_REFRESH,
        created_by=request.user,
        message=f"OneNote refresh queued for {len(request.pages)} page(s).",
    )

    background_tasks.add_task(run_onenote_refresh_job, job["job_id"], request)

    return JobResponse(**job)
