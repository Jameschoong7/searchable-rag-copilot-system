import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "data/simulated/document_metadata.json"


def load_document_metadata() -> list[dict]:
    """Load document metadata records from the current local JSON store."""
    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def save_document_metadata(documents: list[dict]) -> None:
    """Persist document metadata records to the current local JSON store."""
    with METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(documents, metadata_file, indent=2)


def append_document_metadata(new_document: dict) -> None:
    """Append one document metadata record to the current local JSON store."""
    documents = load_document_metadata()
    documents.append(new_document)
    save_document_metadata(documents)


def metadata_exists_for_filename(filename: str) -> bool:
    """Check whether a metadata record already exists for a filename."""
    documents = load_document_metadata()

    return any(
        document["filename"] == filename
        for document in documents
    )


def generate_document_id(documents: list[dict]) -> str:
    """Generate the next local upload document ID."""
    upload_count = sum(
        1 for document in documents
        if document["document_id"].startswith("DOC-UPLOAD-")
    )

    return f"DOC-UPLOAD-{upload_count + 1:03d}"