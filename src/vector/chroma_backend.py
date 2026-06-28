import os
from functools import lru_cache

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
from pathlib import Path


def load_embedding_model() -> HuggingFaceEmbeddings:
    """Load the same embedding model used for local Chroma indexing and retrieval."""
    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL")
    )


@lru_cache(maxsize=1)
def load_vector_store() -> Chroma:
    """Load the cached local Chroma vector store for query-time retrieval."""
    return Chroma(
        persist_directory=os.getenv("CHROMA_DB_PATH"),
        embedding_function=load_embedding_model(),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
    )


def clear_vector_store_cache() -> None:
    """Clear cached Chroma handles before/after rebuild or vector cleanup."""
    load_vector_store.cache_clear()


def similarity_search_with_scores(
    question: str,
    allowed_sources: list[str],
    top_k: int,
) -> list[tuple]:
    """Run Chroma similarity search inside the already-authorized source scope."""
    vector_store = load_vector_store()

    return vector_store.similarity_search_with_relevance_scores(
        question,
        k=top_k,
        filter={"source": {"$in": allowed_sources}},
    )


def store_chunks(chunks: list, db_path: str, collection_name: str) -> Chroma:
    """Create a fresh Chroma collection from chunked documents."""
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=load_embedding_model(),
        collection_name=collection_name,
        persist_directory=db_path,
    )

    print(f"Stored {len(chunks)} chunks to ChromaDB at {db_path}")
    return vector_store


def add_chunks(chunks: list, db_path: str, collection_name: str) -> None:
    """Append chunked documents into an existing Chroma collection."""
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=load_embedding_model(),
        collection_name=collection_name,
    )

    vector_store.add_documents(chunks)


def delete_vectors_for_source(
    source_path: str,
    db_path: str,
    collection_name: str,
) -> int:
    """Delete existing Chroma vectors that came from one source file."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)

    existing = collection.get(
        where={"source": source_path},
        include=[],
    )

    deleted_count = len(existing["ids"])

    if deleted_count:
        collection.delete(ids=existing["ids"])

    return deleted_count


def reset_index(db_path: str, collection_name: str) -> None:
    """Delete the local Chroma index directory before a full rebuild."""
    clear_vector_store_cache()

    chroma_directory = Path(db_path)

    if chroma_directory.exists():
        shutil.rmtree(chroma_directory)


def get_index_record_count() -> int:
    """Count vectors currently stored in the local Chroma collection."""
    db_path = os.getenv("CHROMA_DB_PATH")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME")

    if db_path is None or collection_name is None:
        return 0

    if not Path(db_path).exists():
        return 0

    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return 0

    return collection.count()


def get_index_size_bytes() -> int | None:
    """Calculate local Chroma index folder size in bytes."""
    db_path = os.getenv("CHROMA_DB_PATH")

    if db_path is None:
        return 0

    path = Path(db_path)

    if not path.exists():
        return 0

    return sum(
        file_path.stat().st_size
        for file_path in path.rglob("*")
        if file_path.is_file()
    )