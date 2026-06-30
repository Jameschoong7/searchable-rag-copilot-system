# REQ_F004: Exposes cited RAG answers through an API endpoint
# REQ_F003: Provides the backend route that a future Teams chatbot can call
# REQ_F005: Provides the backend route that the Streamlit web app can call

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
import re


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
    create_job,
    get_job,
    get_latest_job,
    update_job,
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


class UploadValidationResponse(BaseModel):
    """Represent backend-approved upload metadata scope."""

    status: str
    document_department: str
    allowed_roles: list[str]
    allowed_departments: list[str]


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


def run_chat_query_job(job_id: str, request: ChatJobRequest) -> None:
    """Run one chat query in the background and store the answer in the job table."""
    import time

    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        "Retrieving authorised knowledge chunks and generating answer.",
    )

    start_time = time.perf_counter()

    try:
        from src.rag.engine import generate_answer

        result = generate_answer(
            question=request.question.strip(),
            role=request.role,
            department=request.department,
            department_filter=request.department_filter,
            file_type_filter=request.file_type_filter,
        )

        update_job(
            job_id,
            JOB_STATUS_SUCCEEDED,
            "Chat answer generated.",
            {
                "question": result["question"],
                "answer": result["answer"],
                "sources": result["sources"],
                "role": request.role,
                "department": request.department,
                "department_filter": request.department_filter,
                "file_type_filter": request.file_type_filter,
                "latency_seconds": round(time.perf_counter() - start_time, 3),
            },
        )
    except Exception as error:
        update_job(
            job_id,
            JOB_STATUS_FAILED,
            f"Chat query failed: {error}",
            {
                "question": request.question,
                "answer": "",
                "sources": [],
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
    if request.role == GENERAL_EMPLOYEE_ROLE:
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot upload knowledge base documents.",
        )

    if request.role == PROJECT_MANAGER_ROLE:
        return UploadValidationResponse(
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

        return UploadValidationResponse(
            status="approved",
            document_department=request.document_department,
            allowed_roles=expand_allowed_roles(request.allowed_roles),
            allowed_departments=expand_allowed_departments(request.allowed_departments),
        )

    raise HTTPException(
        status_code=403,
        detail="Unknown role cannot upload knowledge base documents.",
      )


@app.post("/admin/archive-document", response_model=ArchiveDocumentResponse)
def archive_document(request: ArchiveDocumentRequest) -> ArchiveDocumentResponse:
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

    return ArchiveDocumentResponse(
        status="success",
        document_id=request.document_id,
        deleted_vector_count=deleted_vector_count,
        message=(
            f"Archived {target_document['title']} and removed "
            f"{deleted_vector_count} vector/index record(s) from the configured backend."
        ),
    )


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