import json
from pathlib import Path

from src.metadata.repository import SEED_METADATA_PATH, load_document_metadata


def export_metadata_seed() -> dict:
    """Export live SQLite metadata into the tracked JSON seed file."""
    documents = load_document_metadata(include_inactive=True)

    SEED_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SEED_METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(documents, metadata_file, indent=2)

    return {
        "status": "success",
        "records_exported": len(documents),
        "path": str(SEED_METADATA_PATH),
    }


if __name__ == "__main__":
    result = export_metadata_seed()
    print(json.dumps(result, indent=2))