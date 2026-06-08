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


class ReindexResponse(BaseModel):
    """Represent the result of a local vector index rebuild."""

    status: str
    documents_indexed: int
    chunks_indexed: int
    message: str


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
def reindex_knowledge_base() -> ReindexResponse:
    """Rebuild the local vector index through the shared backend API."""
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