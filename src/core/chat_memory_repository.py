import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAT_MEMORY_DB_PATH = PROJECT_ROOT / "data/metadata/document_metadata.db"

MESSAGE_ROLE_USER = "user"
MESSAGE_ROLE_ASSISTANT = "assistant"


def now_text() -> str:
    """Return a consistent timestamp string for chat memory records."""
    return datetime.now().isoformat(timespec="seconds")


def initialise_chat_memory_tables() -> None:
    """Create local chat memory tables if they do not exist."""
    CHAT_MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(CHAT_MEMORY_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user TEXT NOT NULL,
                role TEXT NOT NULL,
                department TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
            )
            """
        )


def make_session_title(question: str) -> str:
    """Create a compact session title from the first user question."""
    clean_question = " ".join(question.strip().split())

    if len(clean_question) <= 64:
        return clean_question or "New conversation"

    return f"{clean_question[:61]}..."


def get_chat_session(session_id: str) -> dict | None:
    """Load one chat session by ID."""
    initialise_chat_memory_tables()

    with sqlite3.connect(CHAT_MEMORY_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT
                session_id,
                user,
                role,
                department,
                title,
                created_at,
                updated_at
            FROM chat_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


def create_chat_session(
    user: str,
    role: str,
    department: str,
    first_question: str,
) -> dict:
    """Create a persistent chat session for one authenticated portal user."""
    initialise_chat_memory_tables()

    session_id = str(uuid.uuid4())
    timestamp = now_text()

    with sqlite3.connect(CHAT_MEMORY_DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO chat_sessions (
                session_id,
                user,
                role,
                department,
                title,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user,
                role,
                department,
                make_session_title(first_question),
                timestamp,
                timestamp,
            ),
        )

    return get_chat_session(session_id)


def get_or_create_chat_session(
    session_id: str | None,
    user: str,
    role: str,
    department: str,
    first_question: str,
) -> dict:
    """Continue an owned session, or create a fresh one when needed."""
    if session_id:
        existing_session = get_chat_session(session_id)

        if existing_session and existing_session["user"] == user:
            return existing_session

    return create_chat_session(
        user=user,
        role=role,
        department=department,
        first_question=first_question,
    )


def touch_chat_session(session_id: str) -> None:
    """Refresh the updated timestamp after a new chat turn is stored."""
    initialise_chat_memory_tables()

    with sqlite3.connect(CHAT_MEMORY_DB_PATH) as connection:
        connection.execute(
            """
            UPDATE chat_sessions
            SET updated_at = ?
            WHERE session_id = ?
            """,
            (now_text(), session_id),
        )


def append_chat_message(
    session_id: str,
    message_role: str,
    content: str,
    sources: list[str] | None = None,
    status: str = "",
) -> dict:
    """Append one user or assistant message to a persistent session."""
    initialise_chat_memory_tables()

    message_id = str(uuid.uuid4())
    timestamp = now_text()

    with sqlite3.connect(CHAT_MEMORY_DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (
                message_id,
                session_id,
                message_role,
                content,
                sources_json,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                message_role,
                content,
                json.dumps(sources or []),
                status,
                timestamp,
            ),
        )

    touch_chat_session(session_id)
    return get_chat_message(message_id)


def get_chat_message(message_id: str) -> dict | None:
    """Load one stored chat message by ID."""
    initialise_chat_memory_tables()

    with sqlite3.connect(CHAT_MEMORY_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT
                message_id,
                session_id,
                message_role,
                content,
                sources_json,
                status,
                created_at
            FROM chat_messages
            WHERE message_id = ?
            """,
            (message_id,),
        ).fetchone()

    if row is None:
        return None

    message = dict(row)
    message["sources"] = json.loads(message.pop("sources_json"))
    return message


def list_chat_sessions_for_user(user: str, limit: int = 20) -> list[dict]:
    """Return recent chat sessions owned by one portal user."""
    initialise_chat_memory_tables()

    with sqlite3.connect(CHAT_MEMORY_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                session_id,
                user,
                role,
                department,
                title,
                created_at,
                updated_at
            FROM chat_sessions
            WHERE user = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user, limit),
        ).fetchall()

    return [dict(row) for row in rows]


def list_chat_messages_for_session(session_id: str, user: str) -> list[dict]:
    """Return messages for a session only when it belongs to the requesting user."""
    session = get_chat_session(session_id)

    if session is None or session["user"] != user:
        return []

    with sqlite3.connect(CHAT_MEMORY_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                message_id,
                session_id,
                message_role,
                content,
                sources_json,
                status,
                created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        ).fetchall()

    messages = []

    for row in rows:
        message = dict(row)
        message["sources"] = json.loads(message.pop("sources_json"))
        messages.append(message)

    return messages


def list_recent_chat_messages_for_session(
    session_id: str,
    user: str,
    limit: int,
) -> list[dict]:
    """Return the most recent messages for an owned session in display order."""
    session = get_chat_session(session_id)

    if session is None or session["user"] != user:
        return []

    with sqlite3.connect(CHAT_MEMORY_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                message_id,
                session_id,
                message_role,
                content,
                sources_json,
                status,
                created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    messages = []

    for row in reversed(rows):
        message = dict(row)
        message["sources"] = json.loads(message.pop("sources_json"))
        messages.append(message)

    return messages
