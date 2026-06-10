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
    role: str = "General Employee"
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
    if request.role != "System Admin":
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can rebuild the vector index.",
        )
    try:
        from src.etl.pipeline import rebuild_vector_store

        result = rebuild_vector_store()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Index rebuild failed: {error}",
        ) from error

    return ReindexResponse(
        status="success",
        documents_indexed=result["documents_indexed"],
        chunks_indexed=result["chunks_indexed"],
        message=(
            f"Rebuilt ChromaDB with {result['documents_indexed']} document(s) "
            f"and {result['chunks_indexed']} chunk(s)."
        ),
    )


@app.post("/admin/validate-upload", response_model=UploadValidationResponse)
def validate_upload_metadata(
    request: UploadValidationRequest,
) -> UploadValidationResponse:
    """Validate upload metadata permissions before local file/metadata writes."""
    if request.role == "General Employee":
        raise HTTPException(
            status_code=403,
            detail="General Employee cannot upload knowledge base documents.",
        )

    if request.role == "Project Manager":
        return UploadValidationResponse(
            status="approved",
            document_department=request.user_department,
            allowed_roles=[
                role for role in request.allowed_roles
                if role in ["Project Manager", "General Employee"]
            ] or ["Project Manager"],
            allowed_departments=[request.user_department],
        )

    if request.role == "System Admin":
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
            allowed_roles=request.allowed_roles,
            allowed_departments=request.allowed_departments,
        )

    raise HTTPException(
        status_code=403,
        detail="Unknown role cannot upload knowledge base documents.",
      )