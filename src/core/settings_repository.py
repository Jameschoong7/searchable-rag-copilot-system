import sqlite3
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_DB_PATH = PROJECT_ROOT / "data/metadata/document_metadata.db"


def initialise_settings_table() -> None:
    """Create the runtime settings table inside the local metadata SQLite database."""
    SETTINGS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(SETTINGS_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_by TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_pending_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                requested_by TEXT NOT NULL,
                requested_at TEXT NOT NULL
            )
            """
        )


def load_runtime_settings() -> dict[str, str]:
    """Load saved admin runtime setting overrides from SQLite."""
    initialise_settings_table()

    with sqlite3.connect(SETTINGS_DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT key, value
            FROM app_settings
            ORDER BY key
            """
        ).fetchall()

    return {
        key: value
        for key, value in rows
    }


def save_runtime_settings(
    settings: dict[str, str],
    updated_by: str,
) -> None:
    """Persist admin runtime setting overrides in SQLite."""
    initialise_settings_table()

    updated_at = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(SETTINGS_DB_PATH) as connection:
        for key, value in settings.items():
            connection.execute(
                """
                INSERT INTO app_settings (key, value, updated_by, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                (
                    key,
                    str(value),
                    updated_by,
                    updated_at,
                ),
            )


def load_pending_runtime_settings() -> dict[str, str]:
    """Load risky runtime settings waiting for a successful rebuild."""
    initialise_settings_table()

    with sqlite3.connect(SETTINGS_DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT key, value
            FROM app_pending_settings
            ORDER BY key
            """
        ).fetchall()

    return {
        key: value
        for key, value in rows
    }


def save_pending_runtime_settings(
    settings: dict[str, str],
    requested_by: str,
) -> None:
    """Persist risky runtime settings that should apply only after rebuild succeeds."""
    initialise_settings_table()

    requested_at = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(SETTINGS_DB_PATH) as connection:
        for key, value in settings.items():
            connection.execute(
                """
                INSERT INTO app_pending_settings (key, value, requested_by, requested_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    requested_by = excluded.requested_by,
                    requested_at = excluded.requested_at
                """,
                (
                    key,
                    str(value),
                    requested_by,
                    requested_at,
                ),
            )


def clear_pending_runtime_settings(keys: list[str] | None = None) -> None:
    """Clear pending runtime settings after success or admin cancellation."""
    initialise_settings_table()

    with sqlite3.connect(SETTINGS_DB_PATH) as connection:
        if keys is None:
            connection.execute("DELETE FROM app_pending_settings")
            return

        if not keys:
            return

        placeholders = ", ".join("?" for _ in keys)

        connection.execute(
            f"""
            DELETE FROM app_pending_settings
            WHERE key IN ({placeholders})
            """,
            keys,
        )


def promote_pending_runtime_settings(updated_by: str) -> dict[str, str]:
    """Move pending runtime settings into active settings after rebuild succeeds."""
    pending_settings = load_pending_runtime_settings()

    if not pending_settings:
        return {}

    save_runtime_settings(
        settings=pending_settings,
        updated_by=updated_by,
    )
    clear_pending_runtime_settings()

    return pending_settings