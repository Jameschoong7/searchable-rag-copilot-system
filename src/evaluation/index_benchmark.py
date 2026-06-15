import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from src.metadata.repository import load_document_metadata


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATED_DATA_PATH = PROJECT_ROOT / "data/simulated"
CHROMA_DB_PATH = PROJECT_ROOT / os.getenv("CHROMA_DB_PATH", "data/chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")


def get_directory_size(path: Path) -> int:
    """Calculate total size in bytes for all files inside a directory."""
    if not path.exists():
        return 0

    return sum(
        file_path.stat().st_size
        for file_path in path.rglob("*")
        if file_path.is_file()
    )


def count_simulated_source_files(path: Path) -> int:
    """Count searchable local source files in the simulated document folder."""
    supported_suffixes = {".txt", ".pdf", ".docx"}

    return sum(
        1
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in supported_suffixes
    )


def get_chroma_vector_count() -> int:
    """Count vectors currently stored in the local Chroma collection."""
    if CHROMA_COLLECTION_NAME is None:
        return 0

    if not CHROMA_DB_PATH.exists():
        return 0

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        return 0

    return collection.count()


def build_index_benchmark_snapshot() -> dict:
    """Create one benchmark snapshot for metadata, source files, and ChromaDB."""
    active_documents = load_document_metadata()

    chroma_db_size_bytes = get_directory_size(CHROMA_DB_PATH)

    return {
        "active_metadata_records": len(active_documents),
        "simulated_source_files": count_simulated_source_files(SIMULATED_DATA_PATH),
        "chroma_vector_count": get_chroma_vector_count(),
        "chroma_db_size_bytes": chroma_db_size_bytes,
        "chroma_db_size_mb": round(chroma_db_size_bytes / (1024 * 1024), 2),
    }


if __name__ == "__main__":
    snapshot = build_index_benchmark_snapshot()
    print(json.dumps(snapshot, indent=2))