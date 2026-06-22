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
    "source_document_id",
    "version_number",
    "is_active",
    "content_hash",
    "archived_at",
    "replaced_by_document_id",
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
                visual_extraction_status TEXT NOT NULL,
                source_document_id TEXT,
                version_number INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1,
                content_hash TEXT,
                archived_at TEXT,
                replaced_by_document_id TEXT
            )
            """
        )
    ensure_versioning_columns()


def ensure_versioning_columns() -> None:
    """Add document versioning columns to existing local SQLite metadata stores."""
    versioning_columns = {
        "source_document_id": "TEXT",
        "version_number": "INTEGER DEFAULT 1",
        "is_active": "INTEGER DEFAULT 1",
        "content_hash": "TEXT",
        "archived_at": "TEXT",
        "replaced_by_document_id": "TEXT",
    }

    with sqlite3.connect(METADATA_DB_PATH) as connection:
        existing_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(document_metadata)")
        }

        for column_name, column_type in versioning_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE document_metadata ADD COLUMN {column_name} {column_type}"
                )

        connection.execute(
            """
            UPDATE document_metadata
            SET source_document_id = document_id
            WHERE source_document_id IS NULL
            """
        )


def encode_document_for_sqlite(document: dict) -> dict:
    """Convert list fields into JSON strings before storing a metadata row."""
    encoded_document = document.copy()

    encoded_document["tags_json"] = json.dumps(document.get("tags", []))
    encoded_document["allowed_roles_json"] = json.dumps(document.get("allowed_roles", []))
    encoded_document["allowed_departments_json"] = json.dumps(document.get("allowed_departments", []))
    
    encoded_document["source_document_id"] = document.get(
        "source_document_id",
        document.get("document_id"),
    )
    encoded_document["version_number"] = document.get("version_number", 1)
    encoded_document["is_active"] = document.get("is_active", 1)
    encoded_document["content_hash"] = document.get("content_hash")
    encoded_document["archived_at"] = document.get("archived_at")
    encoded_document["replaced_by_document_id"] = document.get("replaced_by_document_id")

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


def load_document_metadata(include_inactive: bool = False) -> list[dict]:
    """Load document metadata records from the local SQLite metadata store."""
    seed_metadata_database_from_json()

    with sqlite3.connect(METADATA_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        where_clause = "" if include_inactive else "WHERE is_active = 1"

        rows = connection.execute(
            f"""
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
                visual_extraction_status,
                source_document_id,
                version_number,
                is_active,
                content_hash,
                archived_at,
                replaced_by_document_id
            FROM document_metadata
            {where_clause}
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


def update_document_metadata(document_id: str, updated_document: dict) -> None:
    """Update one document metadata record in the local SQLite metadata store."""
    initialise_metadata_database()
    encoded_document = encode_document_for_sqlite(updated_document)

    update_columns = [
        column
        for column in DOCUMENT_COLUMNS
        if column != "document_id"
    ]

    set_clause = ", ".join(
        f"{column} = ?"
        for column in update_columns
    )

    values = [
        encoded_document.get(column)
        for column in update_columns
    ]
    values.append(document_id)

    with sqlite3.connect(METADATA_DB_PATH) as connection:
        connection.execute(
            f"""
            UPDATE document_metadata
            SET {set_clause}
            WHERE document_id = ?
            """,
            values,
        )


def archive_document_version(
    document_id: str,
    replaced_by_document_id: str | None = None,
    archived_at: str | None = None,
) -> None:
    """Mark one document version as archived so it is excluded from normal retrieval."""
    initialise_metadata_database()

    with sqlite3.connect(METADATA_DB_PATH) as connection:
        connection.execute(
            """
            UPDATE document_metadata
            SET
                is_active = 0,
                archived_at = ?,
                replaced_by_document_id = ?
            WHERE document_id = ?
            """,
            (
                archived_at,
                replaced_by_document_id,
                document_id,
            ),
        )


def create_new_document_version(
    previous_document_id: str,
    new_document: dict,
    archived_at: str,
) -> None:
    """Archive the previous version and insert a new active version for the same source document."""
    initialise_metadata_database()

    all_documents = load_document_metadata(include_inactive=True)

    previous_document = next(
        (
            document
            for document in all_documents
            if document["document_id"] == previous_document_id
        ),
        None,
    )

    if previous_document is None:
        raise ValueError(f"Document not found: {previous_document_id}")

    source_document_id = previous_document.get(
        "source_document_id",
        previous_document["document_id"],
    )

    previous_version_number = previous_document.get("version_number") or 1

    versioned_document = new_document.copy()
    versioned_document["source_document_id"] = source_document_id
    versioned_document["version_number"] = previous_version_number + 1
    versioned_document["is_active"] = 1
    versioned_document["archived_at"] = None
    versioned_document["replaced_by_document_id"] = None

    archive_document_version(
        document_id=previous_document_id,
        replaced_by_document_id=versioned_document["document_id"],
        archived_at=archived_at,
    )

    append_document_metadata(versioned_document)


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


def load_pending_index_documents() -> list[dict]:
    """Load active documents that still need vector indexing."""
    pending_values = {"pending", "pending_index"}

    return [
        document
        for document in load_document_metadata()
        if document.get("chunk_id") in pending_values
    ]


def load_replaced_documents_for_new_versions(new_document_ids: list[str]) -> list[dict]:
    """Load archived document versions that were replaced by the given new document IDs."""
    if not new_document_ids:
        return []

    all_documents = load_document_metadata(include_inactive=True)
    new_document_id_set = set(new_document_ids)

    return [
        document
        for document in all_documents
        if (
            document.get("is_active") == 0
            and document.get("replaced_by_document_id") in new_document_id_set
        )
    ]


def mark_documents_indexed(document_ids: list[str]) -> None:
    """Mark documents as indexed after their vectors are updated successfully."""
    if not document_ids:
        return

    initialise_metadata_database()

    placeholders = ", ".join("?" for _ in document_ids)

    with sqlite3.connect(METADATA_DB_PATH) as connection:
        connection.execute(
            f"""
            UPDATE document_metadata
            SET chunk_id = 'indexed'
            WHERE document_id IN ({placeholders})
            """,
            document_ids,
        )


def mark_documents_pending_index(document_ids: list[str]) -> None:
    """Mark documents as needing vector indexing."""
    if not document_ids:
        return

    initialise_metadata_database()

    placeholders = ", ".join("?" for _ in document_ids)

    with sqlite3.connect(METADATA_DB_PATH) as connection:
        connection.execute(
            f"""
            UPDATE document_metadata
            SET chunk_id = 'pending_index'
            WHERE document_id IN ({placeholders})
            """,
            document_ids,
        )


def generate_document_id(documents: list[dict]) -> str:
    """Generate the next local upload document ID."""
    upload_count = sum(
        1 for document in documents
        if document["document_id"].startswith("DOC-UPLOAD-")
    )

    return f"DOC-UPLOAD-{upload_count + 1:03d}"
