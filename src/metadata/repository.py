import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_METADATA_PATH = PROJECT_ROOT / "data/simulated/document_metadata.json"
METADATA_DB_PATH = PROJECT_ROOT / "data/metadata/document_metadata.db"

DOCUMENT_COLUMNS = [
    "document_id",
    "title",
    "filename",
    "file_type",
    "source",
    "department",
    "category",
    "tags_json",
    "allowed_roles_json",
    "allowed_departments_json",
    "uploaded_by",
    "uploaded_at",
    "page_number",
    "chunk_id",
    "visual_extraction_status",
]


def initialise_metadata_database() -> None:
    """Create the local SQLite metadata table if it does not exist."""
    METADATA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(METADATA_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_metadata (
                document_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                filename TEXT NOT NULL UNIQUE,
                file_type TEXT NOT NULL,
                source TEXT NOT NULL,
                department TEXT NOT NULL,
                category TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                allowed_roles_json TEXT NOT NULL,
                allowed_departments_json TEXT NOT NULL,
                uploaded_by TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                page_number TEXT,
                chunk_id TEXT,
                visual_extraction_status TEXT NOT NULL
            )
            """
        )


def encode_document_for_sqlite(document: dict) -> dict:
    """Convert list fields into JSON strings before storing a metadata row."""
    encoded_document = document.copy()

    encoded_document["tags_json"] = json.dumps(document.get("tags", []))
    encoded_document["allowed_roles_json"] = json.dumps(document.get("allowed_roles", []))
    encoded_document["allowed_departments_json"] = json.dumps(document.get("allowed_departments", []))

    encoded_document.pop("tags", None)
    encoded_document.pop("allowed_roles", None)
    encoded_document.pop("allowed_departments", None)

    return encoded_document


def decode_document_from_sqlite(row: sqlite3.Row) -> dict:
    """Convert one SQLite row back into the dict shape used by the app and RAG engine."""
    document = dict(row)

    document["tags"] = json.loads(document.pop("tags_json"))
    document["allowed_roles"] = json.loads(document.pop("allowed_roles_json"))
    document["allowed_departments"] = json.loads(document.pop("allowed_departments_json"))

    return document


def seed_metadata_database_from_json() -> None:
    """Seed SQLite from the existing JSON metadata file when the table is empty."""
    initialise_metadata_database()

    with sqlite3.connect(METADATA_DB_PATH) as connection:
        existing_count = connection.execute(
            "SELECT COUNT(*) FROM document_metadata"
        ).fetchone()[0]

        if existing_count > 0:
            return

    if not SEED_METADATA_PATH.exists():
        return

    with SEED_METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        documents = json.load(metadata_file)

    for document in documents:
        append_document_metadata(document)


def load_document_metadata() -> list[dict]:
    """Load document metadata records from the local SQLite metadata store."""
    seed_metadata_database_from_json()

    with sqlite3.connect(METADATA_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                document_id,
                title,
                filename,
                file_type,
                source,
                department,
                category,
                tags_json,
                allowed_roles_json,
                allowed_departments_json,
                uploaded_by,
                uploaded_at,
                page_number,
                chunk_id,
                visual_extraction_status
            FROM document_metadata
            ORDER BY document_id
            """
        ).fetchall()

    return [
        decode_document_from_sqlite(row)
        for row in rows
    ]


def append_document_metadata(new_document: dict) -> None:
    """Insert one document metadata record into the local SQLite metadata store."""
    initialise_metadata_database()
    encoded_document = encode_document_for_sqlite(new_document)

    placeholders = ", ".join("?" for _ in DOCUMENT_COLUMNS)
    column_names = ", ".join(DOCUMENT_COLUMNS)

    values = [
        encoded_document.get(column)
        for column in DOCUMENT_COLUMNS
    ]

    with sqlite3.connect(METADATA_DB_PATH) as connection:
        connection.execute(
            f"""
            INSERT INTO document_metadata ({column_names})
            VALUES ({placeholders})
            """,
            values,
        )


def metadata_exists_for_filename(filename: str) -> bool:
    """Check whether a metadata record already exists for a filename."""
    seed_metadata_database_from_json()

    with sqlite3.connect(METADATA_DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM document_metadata
            WHERE filename = ?
            LIMIT 1
            """,
            (filename,),
        ).fetchone()

    return row is not None


def generate_document_id(documents: list[dict]) -> str:
    """Generate the next local upload document ID."""
    upload_count = sum(
        1 for document in documents
        if document["document_id"].startswith("DOC-UPLOAD-")
    )

    return f"DOC-UPLOAD-{upload_count + 1:03d}"