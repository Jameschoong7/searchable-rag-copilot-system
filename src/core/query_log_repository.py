import json
import sqlite3
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY_LOG_DB_PATH = PROJECT_ROOT / "data/logs/query_logs.db"
QUERY_HISTORY_LIMIT = 50


def initialise_query_log_database() -> None:
    """Create the shared SQLite query log table if it does not exist."""
    QUERY_LOG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(QUERY_LOG_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user TEXT NOT NULL,
                role TEXT NOT NULL,
                department TEXT NOT NULL,
                question TEXT NOT NULL,
                department_filter TEXT,
                file_type_filter TEXT,
                status TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                latency_seconds REAL NOT NULL
            )
            """
        )

        existing_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(query_logs)")
        }

        optional_columns = {
            "feedback": "TEXT DEFAULT 'none'",
            "feedback_note": "TEXT",
            "feedback_at": "TEXT",
            "answer_text": "TEXT",
            "status_reason": "TEXT",
            "client": "TEXT DEFAULT 'unknown'",
        }

        for column_name, column_type in optional_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE query_logs ADD COLUMN {column_name} {column_type}"
                )


def write_query_log(
    *,
    user: str,
    role: str,
    department: str,
    question: str,
    department_filter: str | None,
    file_type_filter: str | None,
    status: str,
    status_reason: str,
    answer_text: str,
    sources: list[str],
    latency_seconds: float,
    client: str = "unknown",
) -> int:
    """Insert one structured chat query event into the shared SQLite log."""
    initialise_query_log_database()

    with sqlite3.connect(QUERY_LOG_DB_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO query_logs (
                timestamp,
                user,
                role,
                department,
                question,
                department_filter,
                file_type_filter,
                status,
                status_reason,
                answer_text,
                sources_json,
                latency_seconds,
                client
            )
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                user,
                role,
                department,
                question,
                department_filter,
                file_type_filter,
                status,
                status_reason,
                answer_text,
                json.dumps(sources),
                round(latency_seconds, 3),
                client,
            ),
        )

        return cursor.lastrowid


def update_query_feedback(
    query_log_id: int,
    feedback: str,
    feedback_note: str | None = None,
) -> None:
    """Update user feedback for one logged query."""
    initialise_query_log_database()

    with sqlite3.connect(QUERY_LOG_DB_PATH) as connection:
        connection.execute(
            """
            UPDATE query_logs
            SET
                feedback = ?,
                feedback_note = ?,
                feedback_at = ?
            WHERE id = ?
            """,
            (
                feedback,
                feedback_note,
                datetime.now().isoformat(timespec="seconds"),
                query_log_id,
            ),
        )


def read_query_log_summary(limit: int = QUERY_HISTORY_LIMIT) -> dict:
    """Read shared query-log signals for dashboards and advisor recommendations."""
    initialise_query_log_database()

    with sqlite3.connect(QUERY_LOG_DB_PATH) as connection:
        summary_row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_queries,
                COALESCE(AVG(latency_seconds), 0) AS average_latency,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS grounded_answers,
                SUM(CASE WHEN status = 'not_found' THEN 1 ELSE 0 END)
                    AS not_found_queries,
                SUM(CASE WHEN status = 'permission_block' THEN 1 ELSE 0 END)
                    AS permission_blocks,
                SUM(CASE WHEN status IN ('api_error', 'connection_error', 'error') THEN 1 ELSE 0 END)
                    AS error_queries
            FROM query_logs
            """
        ).fetchone()

        recent_outcome_rows = connection.execute(
            """
            SELECT
                timestamp,
                user,
                role,
                department,
                question,
                department_filter,
                file_type_filter,
                status,
                status_reason,
                answer_text,
                sources_json,
                latency_seconds,
                feedback,
                feedback_note
            FROM query_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        recent_rows = connection.execute(
            """
            SELECT
                timestamp,
                user,
                role,
                department,
                question,
                department_filter,
                file_type_filter,
                status,
                latency_seconds
            FROM query_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        daily_latency_rows = connection.execute(
            """
            SELECT
                DATE(timestamp) AS query_date,
                COUNT(*) AS query_count,
                AVG(latency_seconds) AS average_latency
            FROM query_logs
            WHERE DATE(timestamp) >= DATE('now', '-6 days')
            GROUP BY DATE(timestamp)
            ORDER BY DATE(timestamp)
            """
        ).fetchall()

        return {
            "total_queries": summary_row[0],
            "average_latency": summary_row[1],
            "grounded_answers": summary_row[2] or 0,
            "not_found_queries": summary_row[3] or 0,
            "permission_blocks": summary_row[4] or 0,
            "error_queries": summary_row[5] or 0,
            "unresolved_queries": (summary_row[3] or 0) + (summary_row[5] or 0),
            "recent_outcome_rows": recent_outcome_rows,
            "recent_queries": recent_rows,
            "daily_latency_rows": daily_latency_rows,
            "query_history_limit": limit,
        }
