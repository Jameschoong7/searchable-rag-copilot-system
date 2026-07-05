import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOBS_DB_PATH = PROJECT_ROOT / "data/metadata/document_metadata.db"

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"

JOB_TYPE_REINDEX = "reindex"
JOB_TYPE_INDEX_UPDATE = "index_update"
JOB_TYPE_CHAT_QUERY = "chat_query"

JOB_TYPE_ONEDRIVE_STAGE = "onedrive_stage"
JOB_TYPE_ONENOTE_STAGE = "onenote_stage"
JOB_TYPE_DOCUMENT_ARCHIVE = "document_archive"
JOB_TYPE_DOCUMENT_UNARCHIVE = "document_unarchive"

def now_text() -> str:
    """Return a consistent timestamp string for local job records."""
    return datetime.now().isoformat(timespec="seconds")


def initialise_job_table() -> None:
    """Create the local backend job table if it does not exist."""
    JOBS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(JOBS_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def create_job(
    job_type: str,
    created_by: str,
    message: str = "Job queued.",
) -> dict:
    """Create one backend job record and return it."""
    initialise_job_table()

    job_id = str(uuid.uuid4())
    timestamp = now_text()

    with sqlite3.connect(JOBS_DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO app_jobs (
                job_id,
                job_type,
                status,
                message,
                result_json,
                created_by,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_type,
                JOB_STATUS_QUEUED,
                message,
                "{}",
                created_by,
                timestamp,
                timestamp,
            ),
        )

    return get_job(job_id)


def get_job(job_id: str) -> dict | None:
    """Load one backend job record by ID."""
    initialise_job_table()

    with sqlite3.connect(JOBS_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT
                job_id,
                job_type,
                status,
                message,
                result_json,
                created_by,
                created_at,
                updated_at
            FROM app_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

    if row is None:
        return None

    job = dict(row)
    job["result"] = json.loads(job.pop("result_json"))

    return job


def get_latest_job(job_type: str | None = None) -> dict | None:
    """Load the most recently updated backend job."""
    initialise_job_table()

    where_clause = ""
    values = []

    if job_type:
        where_clause = "WHERE job_type = ?"
        values.append(job_type)

    with sqlite3.connect(JOBS_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            f"""
            SELECT
                job_id,
                job_type,
                status,
                message,
                result_json,
                created_by,
                created_at,
                updated_at
            FROM app_jobs
            {where_clause}
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            values,
        ).fetchone()

    if row is None:
        return None

    job = dict(row)
    job["result"] = json.loads(job.pop("result_json"))

    return job


def update_job(
    job_id: str,
    status: str,
    message: str,
    result: dict | None = None,
) -> dict:
    """Update one backend job status and result payload."""
    initialise_job_table()

    with sqlite3.connect(JOBS_DB_PATH) as connection:
        connection.execute(
            """
            UPDATE app_jobs
            SET
                status = ?,
                message = ?,
                result_json = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                status,
                message,
                json.dumps(result or {}),
                now_text(),
                job_id,
            ),
        )

    return get_job(job_id)
