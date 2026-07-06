import hashlib
import hmac
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from src.core.constants import (
    DEPARTMENT_OPTIONS,
    GENERAL_EMPLOYEE_ROLE,
    PROJECT_MANAGER_ROLE,
    SYSTEM_ADMIN_ROLE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
USERS_DB_PATH = PROJECT_ROOT / "data/metadata/document_metadata.db"
PASSWORD_ITERATIONS = 120_000
DEFAULT_SEED_PASSWORD = "password123"


SEED_USERS = [
    {
        "username": "admin_jc",
        "password": DEFAULT_SEED_PASSWORD,
        "role": SYSTEM_ADMIN_ROLE,
        "department": "IT",
    },
    *[
        {
            "username": f"pm_{department.lower()}",
            "password": DEFAULT_SEED_PASSWORD,
            "role": PROJECT_MANAGER_ROLE,
            "department": department,
        }
        for department in DEPARTMENT_OPTIONS
    ],
    *[
        {
            "username": f"employee_{department.lower()}",
            "password": DEFAULT_SEED_PASSWORD,
            "role": GENERAL_EMPLOYEE_ROLE,
            "department": department,
        }
        for department in DEPARTMENT_OPTIONS
    ],
]


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return a PBKDF2 password hash encoded with its salt and iteration count."""
    if salt is None:
        salt = os.urandom(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )

    return (
        f"pbkdf2_sha256${PASSWORD_ITERATIONS}$"
        f"{salt.hex()}${password_hash.hex()}"
    )


def verify_password(password: str, stored_password_hash: str) -> bool:
    """Compare a plaintext password with a stored PBKDF2 password hash."""
    try:
        algorithm, iterations_text, salt_hex, expected_hash_hex = stored_password_hash.split("$", 3)
        iterations = int(iterations_text)
    except ValueError:
        return False

    if algorithm != "pbkdf2_sha256":
        return False

    actual_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        iterations,
    )

    return hmac.compare_digest(actual_hash.hex(), expected_hash_hex)


def initialise_user_table() -> None:
    """Create the local SQLite user table used by the Streamlit portal login."""
    USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(USERS_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                department TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def seed_default_users() -> None:
    """Insert missing preset portal users without overwriting existing accounts."""
    initialise_user_table()

    with sqlite3.connect(USERS_DB_PATH) as connection:
        now = datetime.now().isoformat(timespec="seconds")

        for user in SEED_USERS:
            connection.execute(
                """
                INSERT INTO app_users (
                    username,
                    password_hash,
                    role,
                    department,
                    is_active,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(username) DO NOTHING
                """,
                (
                    user["username"],
                    hash_password(user["password"]),
                    user["role"],
                    user["department"],
                    now,
                    now,
                ),
            )


def get_user(username: str) -> dict | None:
    """Load one user account from SQLite."""
    seed_default_users()

    with sqlite3.connect(USERS_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT username, password_hash, role, department, is_active
            FROM app_users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


def authenticate_user(username: str, password: str) -> dict | None:
    """Return the active user account when username/password authentication succeeds."""
    user = get_user(username)

    if user is None or not user["is_active"]:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    return {
        "username": user["username"],
        "role": user["role"],
        "department": user["department"],
    }
