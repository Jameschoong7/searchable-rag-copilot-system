# REQ_F004: Exposes cited RAG answers through an API endpoint
# REQ_F003: Provides the backend route that a future Teams chatbot can call
# REQ_F005: Provides the backend route that the Streamlit web app can call

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


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


class ReindexResponse(BaseModel):
    """Represent the result of a local vector index rebuild."""

    status: str
    documents_indexed: int
    chunks_indexed: int
    message: str


class IndexUpdatesRequest(BaseModel):
    """Represent an admin request to index pending document updates."""

    role: str


class IndexUpdatesResponse(BaseModel):
    """Represent the result of indexing pending document updates."""

    status: str
    pending_document_count: int
    updated_sources: list[str]
    total_deleted_vectors: int
    total_chunks_indexed: int
    elapsed_seconds: float
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


@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """Answer a user question by calling the shared RAG engine."""
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
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


@app.post("/admin/reindex", response_model=ReindexResponse)
def reindex_knowledge_base(request: ReindexRequest) -> ReindexResponse:
    """Rebuild the local vector index through the shared backend API."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can rebuild the vector index.",
        )
    try:
        import gc
        from src.rag.engine import load_vector_store
        from src.evaluation.index_benchmark import (
            build_full_rebuild_benchmark,
            save_benchmark_result,
        )

        load_vector_store.cache_clear()
        gc.collect()

        benchmark_result = build_full_rebuild_benchmark()
        save_benchmark_result(benchmark_result)
        result = benchmark_result["rebuild_result"]
        
        load_vector_store.cache_clear()
        gc.collect()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Index rebuild failed: {error}",
        ) from error

    return ReindexResponse(
        status="success",
        documents_indexed=result["documents_indexed"],
        document_objects_loaded=result["document_objects_loaded"],
        chunks_indexed=result["chunks_indexed"],
        message=(
            f"Rebuilt ChromaDB with {result['documents_indexed']} file(s), "
            f"{result['document_objects_loaded']} document object(s), "
            f"and {result['chunks_indexed']} chunk(s)."
        ),
    )


@app.post("/admin/index-updates", response_model=IndexUpdatesResponse)
def index_pending_document_updates(request: IndexUpdatesRequest) -> IndexUpdatesResponse:
    """Run incremental indexing for active documents marked as pending index."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can index pending document updates.",
        )

    try:
        import time
        from pathlib import Path

        from src.etl.pipeline import index_changed_documents
        from src.evaluation.index_benchmark import (
            build_index_benchmark_snapshot,
            save_benchmark_result,
        )
        from src.metadata.repository import (
            load_pending_index_documents,
            mark_documents_indexed,
        )

        pending_documents = load_pending_index_documents()

        if not pending_documents:
            return IndexUpdatesResponse(
                status="no_pending_documents",
                pending_document_count=0,
                updated_sources=[],
                total_deleted_vectors=0,
                total_chunks_indexed=0,
                elapsed_seconds=0,
                message="No pending document updates require indexing.",
            )

        source_paths = [
            str(Path("data/simulated") / document["filename"])
            for document in pending_documents
        ]

        before_snapshot = build_index_benchmark_snapshot()

        start_time = time.perf_counter()
        update_result = index_changed_documents(source_paths)
        elapsed_seconds = round(time.perf_counter() - start_time, 3)

        after_snapshot = build_index_benchmark_snapshot()

        benchmark_result = {
            "benchmark_type": "batch_incremental_update",
            "changed_document_count": update_result["changed_document_count"],
            "updated_sources": update_result["updated_sources"],
            "elapsed_seconds": elapsed_seconds,
            "before": before_snapshot,
            "update_results": update_result["update_results"],
            "total_deleted_vectors": update_result["total_deleted_vectors"],
            "total_document_objects_loaded": update_result["total_document_objects_loaded"],
            "total_chunks_indexed": update_result["total_chunks_indexed"],
            "estimated_unchanged_chunks_avoided": max(
                after_snapshot["chroma_vector_count"] - update_result["total_chunks_indexed"],
                0,
            ),
            "after": after_snapshot,
            "delta": {
                "chroma_vector_count": (
                    after_snapshot["chroma_vector_count"]
                    - before_snapshot["chroma_vector_count"]
                ),
                "chroma_db_size_bytes": (
                    after_snapshot["chroma_db_size_bytes"]
                    - before_snapshot["chroma_db_size_bytes"]
                ),
                "chroma_db_size_mb": round(
                    after_snapshot["chroma_db_size_mb"]
                    - before_snapshot["chroma_db_size_mb"],
                    2,
                ),
            },
        }

        save_benchmark_result(benchmark_result)

        mark_documents_indexed(
            [
                document["document_id"]
                for document in pending_documents
            ]
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Pending index update failed: {error}",
        ) from error

    return IndexUpdatesResponse(
        status="success",
        pending_document_count=len(pending_documents),
        updated_sources=update_result["updated_sources"],
        total_deleted_vectors=update_result["total_deleted_vectors"],
        total_chunks_indexed=update_result["total_chunks_indexed"],
        elapsed_seconds=elapsed_seconds,
        message=(
            f"Indexed {len(pending_documents)} pending document(s), refreshed "
            f"{update_result['total_chunks_indexed']} chunk(s), and replaced "
            f"{update_result['total_deleted_vectors']} old vector(s)."
        ),
    )


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