# REQ_F005: Streamlit Admin Web Portal for manager/admin workflows
# REQ_F004: Displays cited answers returned by the shared FastAPI RAG backend

from html import escape
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time
import requests
import streamlit as st
import altair as alt
import pandas as pd
from dotenv import load_dotenv
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.metadata.repository import (
    load_document_metadata,
    update_document_metadata,
)


from src.core.constants import (
    FILTER_ALL,
    DEPARTMENT_OPTIONS,
    GENERAL_EMPLOYEE_ROLE,
    PROJECT_MANAGER_ROLE,
    ROLE_OPTIONS,
    SYSTEM_ADMIN_ROLE,
)


ROLE_AWARE_CHAT_PROMPTS = {
    SYSTEM_ADMIN_ROLE: {
        "Password Policy": "What are the password policy requirements?",
        "Security Incident": "What is the security incident reporting procedure?",
        "Annual Leave": "What is the annual leave approval process?",
        "VPN Setup": "How do I set up the company VPN?",
    },
    PROJECT_MANAGER_ROLE: {
        "Development Workflow": "What is the software development workflow?",
        "Coding Standards": "What are the Python coding standards?",
        "VPN Setup": "How do I set up the company VPN?",
        "Security Incident": "What is the security incident reporting procedure?",
    },
    GENERAL_EMPLOYEE_ROLE: {
        "Annual Leave": "What is the annual leave approval process?",
        "Expense Claims": "How do I submit an expense claim?",
        "Security Incident": "What is the security incident reporting procedure?",
        "Onboarding": "What is the employee onboarding process?",
    },
}

DEMO_ACCOUNTS = {
    "admin_jc": {
        "password": "password123",
        "role": SYSTEM_ADMIN_ROLE,
        "department": "IT",
    },
    "pm_engineering": {
        "password": "password123",
        "role": PROJECT_MANAGER_ROLE,
        "department": "Engineering",
    },
    "employee_hr": {
        "password": "password123",
        "role": GENERAL_EMPLOYEE_ROLE,
        "department": "HR",
    },
}


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

UPLOAD_VALIDATE_URL = f"{API_BASE_URL}/admin/validate-upload"
REINDEX_URL = f"{API_BASE_URL}/admin/reindex"
API_URL = f"{API_BASE_URL}/query"
API_HEALTH_URL = f"{API_BASE_URL}/health"
CHAT_JOBS_URL = f"{API_BASE_URL}/chat/jobs"
BACKEND_JOBS_URL = f"{API_BASE_URL}/admin/jobs"
REINDEX_JOBS_URL = f"{API_BASE_URL}/admin/reindex-jobs"
INDEX_UPDATE_JOBS_URL = f"{API_BASE_URL}/admin/index-update-jobs"
METADATA_UPDATE_VALIDATE_URL = f"{API_BASE_URL}/admin/validate-metadata-update"
ARCHIVE_DOCUMENT_URL = f"{API_BASE_URL}/admin/archive-document"
SETTINGS_URL = f"{API_BASE_URL}/admin/settings"
UPLOAD_DOCUMENT_URL = f"{API_BASE_URL}/admin/upload-document"
UPLOAD_DOCUMENT_VERSION_URL = f"{API_BASE_URL}/admin/upload-document-version"
APPROVE_DOCUMENT_URL = f"{API_BASE_URL}/admin/approve-document"
QUERY_LOG_DB_PATH = PROJECT_ROOT / "data/logs/query_logs.db"
ONEDRIVE_FILES_URL = f"{API_BASE_URL}/admin/graph/onedrive/files"
ONEDRIVE_STAGE_FILE_URL = f"{API_BASE_URL}/admin/graph/onedrive/stage-file"
EVALUATION_RESULTS_PATH = PROJECT_ROOT / "data/evaluation/retrieval_eval_results.json"
INDEX_BENCHMARK_RESULTS_PATH = PROJECT_ROOT / "data/evaluation/index_benchmark_results.json"
INDEX_BENCHMARK_HISTORY_PATH = PROJECT_ROOT / "data/evaluation/index_benchmark_history.json"
QUERY_HISTORY_LIMIT = 50


def request_document_approval(
    document_id: str,
    title: str,
    department: str,
    category: str,
    tags: list[str],
    allowed_roles: list[str],
    allowed_departments: list[str],
) -> dict:
    """Ask FastAPI to approve one pending-review connector document."""
    response = requests.post(
        APPROVE_DOCUMENT_URL,
        json={
            "role": st.session_state["role"],
            "user_department": st.session_state["department"],
            "document_id": document_id,
            "title": title,
            "department": department,
            "category": category,
            "tags": tags,
            "allowed_roles": allowed_roles,
            "allowed_departments": allowed_departments,
        },
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def request_onedrive_file_scan() -> dict:
    """Ask FastAPI to list files under the configured OneDrive connector root."""
    response = requests.post(
        ONEDRIVE_FILES_URL,
        json={
            "role": st.session_state["role"],
        },
        timeout=60,
    )

    response.raise_for_status()
    return response.json()


def request_onedrive_file_stage(file_item: dict) -> dict:
    """Ask FastAPI to download and stage one OneDrive file for metadata review."""
    response = requests.post(
        ONEDRIVE_STAGE_FILE_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "item_id": file_item["id"],
            "name": file_item["name"],
            "connector_path": file_item["connector_path"],
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def request_backend_reindex() -> dict:
    """Ask the FastAPI backend to rebuild the configured search index."""
    response = requests.post(
        REINDEX_URL,
        json={
            "role": st.session_state["role"],
        },
        timeout=900,
    )

    response.raise_for_status()
    return response.json()


def submit_reindex_job() -> dict:
    """Submit a full search-index rebuild as a durable backend job."""
    response = requests.post(
        REINDEX_JOBS_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def submit_index_update_job() -> dict:
    """Submit pending-document indexing as a durable backend job."""
    response = requests.post(
        INDEX_UPDATE_JOBS_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def request_admin_settings() -> dict:
    """Load backend-owned runtime settings from FastAPI."""
    response = requests.get(
        SETTINGS_URL,
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def request_settings_update(settings: dict[str, str]) -> dict:
    """Ask FastAPI to validate and save runtime settings."""
    response = requests.post(
        SETTINGS_URL,
        json={
            "role": st.session_state["role"],
            "updated_by": st.session_state["user"],
            "settings": settings,
        },
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def request_document_archive(document_id: str) -> dict:
    """Ask FastAPI to archive one document and remove its vectors."""
    response = requests.post(
        ARCHIVE_DOCUMENT_URL,
        json={
            "role": st.session_state["role"],
            "user_department": st.session_state["department"],
            "document_id": document_id,
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def request_upload_validation(
    document_department: str,
    allowed_roles: list[str],
    allowed_departments: list[str],
) -> dict:
    """Ask FastAPI to validate upload metadata for the current user."""
    response = requests.post(
        UPLOAD_VALIDATE_URL,
        json={
            "role": st.session_state["role"],
            "user_department": st.session_state["department"],
            "document_department": document_department,
            "allowed_roles": allowed_roles,
            "allowed_departments": allowed_departments,
        },
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def request_metadata_update_validation(
    document_department: str,
    allowed_roles: list[str],
    allowed_departments: list[str],
) -> dict:
    """Ask FastAPI to validate metadata edits for the current user."""
    response = requests.post(
        METADATA_UPDATE_VALIDATE_URL,
        json={
            "role": st.session_state["role"],
            "user_department": st.session_state["department"],
            "document_department": document_department,
            "allowed_roles": allowed_roles,
            "allowed_departments": allowed_departments,
        },
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def request_backend_document_upload(
    uploaded_file,
    title: str,
    document_department: str,
    category: str,
    tags: list[str],
    allowed_roles: list[str],
    allowed_departments: list[str],
) -> dict:
    """Ask FastAPI to save an uploaded document and create trusted metadata."""
    response = requests.post(
        UPLOAD_DOCUMENT_URL,
        data={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "user_department": st.session_state["department"],
            "title": title,
            "document_department": document_department,
            "category": category,
            "tags_json": json.dumps(tags),
            "allowed_roles_json": json.dumps(allowed_roles),
            "allowed_departments_json": json.dumps(allowed_departments),
        },
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json()


def request_backend_document_version_upload(
    uploaded_file,
    previous_document_id: str,
) -> dict:
    """Ask FastAPI to save a replacement file and create a new document version."""
    response = requests.post(
        UPLOAD_DOCUMENT_VERSION_URL,
        data={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "user_department": st.session_state["department"],
            "previous_document_id": previous_document_id,
        },
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json()


def ask_backend(
    question: str,
    department_filter: str | None,
    file_type_filter: str | None,
) -> dict:
    """Send one user question and session context to the shared FastAPI RAG backend."""
    response = requests.post(
        API_URL,
        json={
            "question": question,
            "role": st.session_state["role"],
            "department": st.session_state["department"],
            "department_filter": department_filter,
            "file_type_filter": file_type_filter,
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json()


def submit_chat_job(
    question: str,
    department_filter: str | None,
    file_type_filter: str | None,
) -> dict:
    """Submit one chat question as a durable backend job."""
    response = requests.post(
        CHAT_JOBS_URL,
        json={
            "user": st.session_state["user"],
            "question": question,
            "role": st.session_state["role"],
            "department": st.session_state["department"],
            "department_filter": department_filter,
            "file_type_filter": file_type_filter,
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def get_backend_job(job_id: str) -> dict:
    """Poll one backend job by ID."""
    response = requests.get(
        f"{BACKEND_JOBS_URL}/{job_id}",
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


@st.fragment(run_every="2s")
def poll_active_chat_job() -> None:
    """Poll the active chat job without rerunning the whole Streamlit app aggressively."""
    active_chat_job_id = st.session_state.get("active_chat_job_id")

    if not active_chat_job_id:
        return

    try:
        job = get_backend_job(active_chat_job_id)
    except requests.exceptions.RequestException as error:
        st.warning(f"Chat job status unavailable: {error}")
        return

    if job["status"] in ["queued", "running"]:
        st.info(job["message"])
        return

    if job["status"] == "succeeded":
        result = job["result"]
        answer_status = classify_answer_status(
            result["answer"],
            result["sources"],
        )

        query_log_id = write_query_log(
            question=result["question"],
            department_filter=result.get("department_filter"),
            file_type_filter=result.get("file_type_filter"),
            status=answer_status,
            sources=result["sources"],
            latency_seconds=result.get("latency_seconds", 0),
        )

        context_text = (
            f"Access context: {result['role']} / {result['department']} | "
            f"Search department: {result.get('department_filter') or 'ACL-permitted shared scope'} | "
            f"File type: {result.get('file_type_filter')}"
        )

        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "context": context_text,
                "status": answer_status,
                "query_log_id": query_log_id,
                "feedback": "none",
            }
        )

    elif job["status"] == "failed":
        result = job.get("result", {})

        query_log_id = write_query_log(
            question=result.get("question", "Unknown question"),
            department_filter=result.get("department_filter"),
            file_type_filter=result.get("file_type_filter"),
            status="api_error",
            sources=[],
            latency_seconds=result.get("latency_seconds", 0),
        )

        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": job["message"],
                "sources": [],
                "context": "",
                "status": "api_error",
                "query_log_id": query_log_id,
                "feedback": "none",
            }
        )

    st.session_state.pop("active_chat_job_id", None)
    st.session_state["chat_is_processing"] = False
    st.rerun()


@st.fragment(run_every="2s")
def poll_active_reindex_job() -> None:
    """Poll the active reindex job without blocking the whole Streamlit app."""
    active_reindex_job_id = st.session_state.get("active_reindex_job_id")

    if not active_reindex_job_id:
        return

    try:
        job = get_backend_job(active_reindex_job_id)
    except requests.exceptions.RequestException as error:
        st.warning(f"Reindex job status unavailable: {error}")
        return

    if job["status"] in ["queued", "running"]:
        st.info(job["message"])
        return

    if job["status"] == "succeeded":
        result = job["result"]
        st.session_state["settings_message"] = result["message"]
        st.session_state["settings_rebuild_required"] = False
        st.session_state["reindex_job_message"] = result["message"]
        st.session_state["reindex_job_status"] = "success"

    elif job["status"] == "failed":
        st.session_state["settings_message"] = job["message"]
        st.session_state["settings_rebuild_required"] = True
        st.session_state["reindex_job_message"] = job["message"]
        st.session_state["reindex_job_status"] = "error"

    st.session_state.pop("active_reindex_job_id", None)
    st.rerun()


@st.fragment(run_every="2s")
def poll_active_index_update_job() -> None:
    """Poll the active pending-index update job without blocking Streamlit navigation."""
    active_index_update_job_id = st.session_state.get("active_index_update_job_id")

    if not active_index_update_job_id:
        return

    try:
        job = get_backend_job(active_index_update_job_id)
    except requests.exceptions.RequestException as error:
        st.warning(f"Index update job status unavailable: {error}")
        return

    if job["status"] in ["queued", "running"]:
        st.info(job["message"])
        return

    if job["status"] == "succeeded":
        result = job["result"]
        st.session_state["index_update_job_message"] = result["message"]
        st.session_state["index_update_job_status"] = "success"

    elif job["status"] == "failed":
        st.session_state["index_update_job_message"] = job["message"]
        st.session_state["index_update_job_status"] = "error"

    st.session_state.pop("active_index_update_job_id", None)
    st.rerun()


def is_api_online() -> bool:
    """Check whether the lightweight FastAPI health endpoint is reachable."""
    try:
        response = requests.get(API_HEALTH_URL, timeout=3)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return False

    return True


def select_example_chat_prompt(prompt: str) -> None:
    """Place an example prompt into the chat box so the user can review it."""
    st.session_state["chat_question"] = prompt


def submit_chat_question() -> None:
    """Stage the submitted draft and clear the visible input before rerun."""
    draft_question = st.session_state.get("chat_question", "")

    st.session_state["pending_chat_question"] = draft_question
    st.session_state["chat_question"] = ""
    st.session_state["chat_is_processing"] = bool(draft_question.strip())


def is_logged_in() -> bool:
    """Check whether the current Streamlit session has an authenticated demo user."""
    return "user" in st.session_state


def login_user(username: str, password: str) -> bool:
    """Validate a demo account and store role/department in the session."""
    account = DEMO_ACCOUNTS.get(username)

    if account is None or account["password"] != password:
        return False

    st.session_state["user"] = username
    st.session_state["role"] = account["role"]
    st.session_state["department"] = account["department"]
    return True


def logout_user() -> None:
    """Clear demo authentication state from the current Streamlit session."""
    st.session_state.clear()


def get_kb_page_label() -> str:
    """Return the KB page label based on the current user's role."""
    if st.session_state["role"] in [SYSTEM_ADMIN_ROLE, PROJECT_MANAGER_ROLE]:
        return "KB Management"

    return "KB Status"


def can_access_settings() -> bool:
    """Check whether the current user can access admin-only settings."""
    return st.session_state["role"] == SYSTEM_ADMIN_ROLE


def load_retrieval_evaluation_results() -> dict | None:
    """Load the latest local retrieval evaluation output if it exists."""
    if not EVALUATION_RESULTS_PATH.exists():
        return None

    with EVALUATION_RESULTS_PATH.open("r", encoding="utf-8") as results_file:
        return json.load(results_file)


def load_index_benchmark_results() -> dict | None:
    """Load the latest local vector index benchmark output if it exists."""
    if not INDEX_BENCHMARK_RESULTS_PATH.exists():
        return None

    with INDEX_BENCHMARK_RESULTS_PATH.open("r", encoding="utf-8") as results_file:
        return json.load(results_file)


def load_index_benchmark_history() -> list[dict]:
    """Load benchmark history so dashboard comparisons use measured baselines."""
    if not INDEX_BENCHMARK_HISTORY_PATH.exists():
        return []

    with INDEX_BENCHMARK_HISTORY_PATH.open("r", encoding="utf-8") as history_file:
        return json.load(history_file)


def initialise_query_log_database() -> None:
    """Create the local SQLite query log table if it does not exist."""
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

        feedback_columns = {
            "feedback": "TEXT DEFAULT 'none'",
            "feedback_note": "TEXT",
            "feedback_at": "TEXT",
        }

        for column_name, column_type in feedback_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE query_logs ADD COLUMN {column_name} {column_type}"
                )


def classify_answer_status(answer: str, sources: list[str]) -> str:
    """Classify the result so dashboard metrics can group query outcomes."""
    lowered_answer = answer.lower()

    if "insufficient permission" in lowered_answer:
        return "permission_block"
    
    not_found_phrases = [
        "not explicitly stated",
        "unable to find",
        "information missing",
        "could not find",
        "not found",
    ]

    if any(phrase in lowered_answer for phrase in not_found_phrases):
        return "not_found"

    if not sources:
        return "not_found"
    
    return "success"


def get_status_label(status: str) -> str:
    """Return a user-facing label for one chat answer status"""
    labels = {
          "success": "Grounded Answer",
          "permission_block": "Permission Block",
          "not_found": "Not Found",
          "api_error": "API Error",
          "connection_error": "Connection Error",
    }

    return labels.get(status, "System Response")


def show_status_message(status: str) -> None:
    """Render a visible chat result state for the current assistant answer."""
    label = get_status_label(status)

    if status == "success":
        st.success(label)
    elif status == "permission_block":
        st.warning(label)
    elif status == "not_found":
        st.info(label)
    else:
        st.error(label)


def show_escalation_guidance(status: str) -> None:
    """Show next-step guidance when the answer is blocked or unresolved."""
    if status == "permission_block":
        st.caption(
            "Next step: request access from your project manager, document owner, "
            "or the department that owns this policy. The system did not use restricted "
            "content to generate the answer."
        )
    elif status == "not_found":
        st.caption(
            "Next step: ask a more specific question, check whether another department owns "
            "this information, ask your manager, or request that an admin adds the missing "
            "document to the knowledge base."
        )
    elif status in ["api_error", "connection_error"]:
        st.caption(
            "Next step: retry later or contact the system admin if the issue continues."
        )


def write_query_log(
        question: str,
        department_filter: str | None,
        file_type_filter: str | None,
        status: str,
        sources: list[str],
        latency_seconds: float,
) -> int:
    """Insert one structured chat query event into the local SQLite log."""
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
                sources_json,
                latency_seconds
            )
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                st.session_state["user"],
                st.session_state["role"],
                st.session_state["department"],
                question,
                department_filter,
                file_type_filter,
                status,
                json.dumps(sources),
                round(latency_seconds, 3),
            )
        )

        return cursor.lastrowid


def update_query_feedback(query_log_id: int, feedback: str, feedback_note: str | None = None) -> None:
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


def read_query_log_summary() -> dict:
    """Read real local query-log signals for the Performance dashboard."""
    initialise_query_log_database()

    with sqlite3.connect(QUERY_LOG_DB_PATH) as connection:
        summary_row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_queries,
                COALESCE(AVG(latency_seconds), 0) AS average_latency,
                SUM(CASE WHEN status = 'permission_block' THEN 1 ELSE 0 END)
                    AS permission_blocks,
                SUM(CASE WHEN status IN ('not_found', 'error') THEN 1 ELSE 0 END)
                    AS unresolved_queries
            FROM query_logs
            """
        ).fetchone()

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
            (QUERY_HISTORY_LIMIT,),
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

        review_rows = connection.execute(
            """
            SELECT
                id,
                timestamp,
                user,
                role,
                department,
                question,
                status,
                sources_json,
                latency_seconds,
                feedback,
                feedback_note
            FROM query_logs
            WHERE
                status != 'success'
                OR feedback = 'reported_issue'
            ORDER BY id DESC
            LIMIT ?
            """,
            (QUERY_HISTORY_LIMIT,),
        ).fetchall()

        return {
            "total_queries": summary_row[0],
            "average_latency": summary_row[1],
            "permission_blocks": summary_row[2] or 0,
            "unresolved_queries": summary_row[3] or 0,
            "recent_queries": recent_rows,
            "daily_latency_rows": daily_latency_rows,
            "query_history_limit": QUERY_HISTORY_LIMIT,
            "review_rows": review_rows,
        }


def can_view_document(document: dict) -> bool:
    """Check whether the current user can see a document metadata row."""
    role = st.session_state["role"]
    department = st.session_state["department"]

    if role == SYSTEM_ADMIN_ROLE:
        return True

    return (
        role in document["allowed_roles"]
        and department in document["allowed_departments"]
    )


def get_index_status_label(document: dict) -> str:
    """Return a readable indexing status for KB Management."""
    chunk_id = document.get("chunk_id")

    if document.get("is_active") == 0:
        return "Archived"

    if chunk_id in ["pending", "pending_index"]:
        return "Pending Index"

    if chunk_id == "indexed":
        return "Indexed"

    return "Unknown"


def get_version_label(document: dict) -> str:
    """Return a readable document version label."""
    version_number = document.get("version_number") or 1

    return f"v{version_number}"


def normalise_uploaded_filename(filename: str) -> str:
    """Return the local filename used for uploaded simulated documents."""
    return filename.replace(" ", "_")


def prepare_upload_title_state(uploaded_file, upload_form_version: int) -> str:
    """Prepare an editable title field when a new TXT file is selected."""
    title_key = f"upload_title{upload_form_version}"
    filename_key = f"upload_filename{upload_form_version}"

    if uploaded_file is None:
        return title_key

    current_filename = normalise_uploaded_filename(uploaded_file.name)
    previous_filename = st.session_state.get(filename_key)

    if previous_filename != current_filename:
        st.session_state[filename_key] = current_filename
        st.session_state[title_key] = infer_title_from_uploaded_file(uploaded_file)

    return title_key


def infer_title_from_uploaded_file(uploaded_file) -> str:
    """Infer a default document title from the uploaded filename."""
    return Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ").title()


st.set_page_config(
    page_title="Searchable RAG Copilot",
    layout="wide",
)


if not is_logged_in():
    st.title("Searchable RAG Copilot")
    st.caption("Sign in to the standalone Admin Web Portal.")
    
    with st.container(border=True):
        username = st.text_input("Username", value="admin_jc")
        password = st.text_input("Password", value="password123", type="password")

        if st.button("Sign In", type="primary"):
            if login_user(username.strip(), password):
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.stop()


api_online = is_api_online()
api_status_label = "API Online" if api_online else "API Offline"
api_status_color = "#166534" if api_online else "#991b1b"
api_status_background = "#dcfce7" if api_online else "#fee2e2"

# Global CSS updates for polished dashboard cards
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] {{
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] {{
        gap: 0.2rem;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label {{
        width: 100%;
        padding: 0.58rem 0.7rem;
        border-radius: 0.3rem;
        cursor: pointer;
        transition: background-color 0.15s ease;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background-color: #eef2f7;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
        background-color: #e2e8f0;
        font-weight: 600;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] > label
    [data-testid="stMarkdownContainer"] {{
        width: 100%;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {{
        display: none;
    }}

    [data-testid="stForm"] [data-testid="InputInstructions"] {{
        display: none;
    }}
    
    div[data-testid="stMetric"] {{
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }}
    </style>
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0 0.65rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    ">
        <div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">
                Searchable RAG Copilot
            </div>
            <div style="font-size: 0.85rem; color: #64748b;">
                Enterprise Knowledge Portal
            </div>
        </div>
        <div style="
            color: {api_status_color};
            background: {api_status_background};
            border-radius: 0.3rem;
            padding: 0.28rem 0.55rem;
            font-size: 0.76rem;
            font-weight: 600;
            border: 1px solid {api_status_color}30;
        ">
            {escape(api_status_label)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Account")
st.sidebar.caption(escape(st.session_state["user"]))
st.sidebar.markdown(
    f"""
    <div style="
        padding: 0.75rem;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        background: #ffffff;
        font-size: 0.85rem;
        line-height: 1.65;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    ">
        <strong style="color: #475569;">Role</strong><br>
        <span style="font-weight: 500; color: #0f172a;">{escape(st.session_state["role"])}</span><br>
        <strong style="color: #475569; display: inline-block; margin-top: 0.25rem;">Department</strong><br>
        <span style="font-weight: 500; color: #0f172a;">{escape(st.session_state["department"])}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

poll_active_reindex_job()
poll_active_index_update_job()

kb_page_label = get_kb_page_label()

page_options = [
    kb_page_label,
    "Chat",
]

if st.session_state["role"] in [SYSTEM_ADMIN_ROLE, PROJECT_MANAGER_ROLE]:
    page_options.insert(0, "Performance")

if can_access_settings():
    page_options.append("Settings")

selected_page = st.sidebar.radio("Navigation", page_options)

st.sidebar.divider()

if st.sidebar.button("Logout", use_container_width=True):
    logout_user()
    st.rerun()


if st.session_state["role"] == SYSTEM_ADMIN_ROLE and selected_page != kb_page_label:
    pending_index_documents = [
        document
        for document in load_document_metadata()
        if get_index_status_label(document) == "Pending Index"
    ]

    if pending_index_documents:
        with st.container(border=True):
            st.warning(
                f"**Admin Action Required:** {len(pending_index_documents)} document(s) waiting for search index update."
            )
            
            st.markdown("Review pending documents in **KB Management**, then run **Update for Pending Documents**.")

            with st.expander("View waiting documents"):
                for document in pending_index_documents[:3]:
                    st.caption(f"• {document['title']} ({document['document_id']})")
                
                if len(pending_index_documents) > 3:
                    st.caption(f"+ {len(pending_index_documents) - 3} more...")


if selected_page == "Performance":
    st.header("Performance Dashboard")
    st.caption(
        "Monitoring real local queries, retrieval evaluation metrics, "
        "and active vector index benchmarks."
    )

    documents = load_document_metadata()
    indexed_document_count = len(documents)
    query_log_summary = read_query_log_summary()
    evaluation_results = load_retrieval_evaluation_results()
    index_benchmark_results = load_index_benchmark_results()
    index_benchmark_history = load_index_benchmark_history()
    latest_full_rebuild_result = next(
        (
            result for result in reversed(index_benchmark_history)
            if result.get("benchmark_type") == "full_rebuild"
        ),
        None,
    )

    if evaluation_results:
        evaluation_summary = evaluation_results["summary"]["overall"]
        top_k_accuracy = f"{evaluation_summary['top_k_accuracy_percent']}%"
        miss_rate_value = (
            evaluation_summary["miss_count"] / evaluation_summary["total_queries"]
            if evaluation_summary["total_queries"]
            else 0
        )
        miss_rate = f"{miss_rate_value * 100:.1f}%"
        top_k_delta = (
            f"{evaluation_summary['correct_queries']} / "
            f"{evaluation_summary['total_queries']} labelled queries"
        )
        miss_delta = f"{evaluation_summary['miss_count']} misses"
    else:
        top_k_accuracy = "No run"
        miss_rate = "No run"
        top_k_delta = "Run retrieval evaluation"
        miss_delta = "Awaiting labelled result"

    with st.container(border=True):
        st.subheader("System Metrics")

        metric_columns = st.columns(4)

        with metric_columns[0]:
            st.metric(
                "Time-to-First-Answer",
                f"{query_log_summary['average_latency']:.2f}s",
                "Local logged average",
            )

        with metric_columns[1]:
            st.metric(
                "Eval Top-K Accuracy (K=5)",
                top_k_accuracy,
                top_k_delta,
            )

        with metric_columns[2]:
            st.metric(
                "Miss Rate",
                miss_rate,
                miss_delta,
            )

        with metric_columns[3]:
            st.metric(
                "Indexed Documents",
                f"{indexed_document_count}",
                "Active KB records",
            )

        st.caption(
            "TTFA is calculated from local chat logs. Eval Top-K Accuracy and Miss Rate come "
            "from the latest labelled retrieval evaluation run. Live chat retrieval uses "
            "the active Top-K and threshold from Settings."
        )

    with st.container(border=True):
        st.subheader("Live Query Signals")

        live_metric_columns = st.columns(3)

        with live_metric_columns[0]:
            st.metric(
                "Logged Queries",
                query_log_summary["total_queries"]
            )

        with live_metric_columns[1]:
            st.metric(
                "Permission Blocks",
                query_log_summary["permission_blocks"],
            )

        with live_metric_columns[2]:
            st.metric(
                "Not Found / Errors",
                query_log_summary["unresolved_queries"],
            )

    st.subheader("Search Index Update Status")
    st.caption(
        "Shows whether the active search index reflects approved/latest documents, "
        "and compares full rebuild work against incremental document updates."
    )

    if index_benchmark_results:
        after_snapshot = index_benchmark_results.get("after", index_benchmark_results)
        benchmark_type = index_benchmark_results.get("benchmark_type", "snapshot")

        active_vectors = after_snapshot.get("indexed_chunk_count", after_snapshot.get("chroma_vector_count", 0))
        active_records = after_snapshot["active_metadata_records"]
        physical_files = after_snapshot["simulated_source_files"]
        db_size_mb = after_snapshot.get("index_size_mb", after_snapshot.get("chroma_db_size_mb"))
        vector_backend = after_snapshot.get("vector_backend", "chroma")
        archived_file_count = max(physical_files - active_records, 0)

        if benchmark_type == "batch_incremental_update":
            changed_document_count = index_benchmark_results["changed_document_count"]
            chunks_refreshed = index_benchmark_results["total_chunks_indexed"]
            deleted_vectors = index_benchmark_results["total_deleted_vectors"]
            avoided_chunks = index_benchmark_results["estimated_unchanged_chunks_avoided"]
            elapsed_seconds = index_benchmark_results["elapsed_seconds"]
            before_snapshot = index_benchmark_results["before"]
            before_active_vectors = before_snapshot.get(
                "indexed_chunk_count",
                before_snapshot.get("chroma_vector_count", 0),
            )

            document_label = (
                "document"
                if changed_document_count == 1
                else "documents"
            )

            st.success(
                f"Latest index update processed {changed_document_count} changed "
                f"{document_label} in {elapsed_seconds}s. {chunks_refreshed} chunks were "
                f"re-indexed and {avoided_chunks} unchanged chunks were skipped."
            )

            metric_columns = st.columns(4)

            with metric_columns[0]:
                st.metric(
                    "Changed Documents",
                    changed_document_count,
                    "Latest update run",
                )

            with metric_columns[1]:
                st.metric(
                    "Chunks Re-indexed",
                    chunks_refreshed,
                    f"{avoided_chunks} unchanged avoided",
                )

            with metric_columns[2]:
                st.metric(
                    "Old Vectors Removed",
                    deleted_vectors,
                    "Removed before re-index",
                )

            with metric_columns[3]:
                active_index_delta = (
                "Portal only"
                if db_size_mb is None
                else f"{db_size_mb} MB"
            )

                st.metric(
                    "Active Index",
                    f"{active_vectors} vectors",
                    active_index_delta,
                )

            avoided_percent = (
                avoided_chunks / before_active_vectors * 100
                if before_active_vectors
                else 0
            )
            refreshed_percent = (
                chunks_refreshed / before_active_vectors
                if before_active_vectors
                else 0
            )
            if latest_full_rebuild_result:
                full_rebuild_baseline_seconds = latest_full_rebuild_result["elapsed_seconds"]
                full_rebuild_baseline_chunks = latest_full_rebuild_result["rebuild_result"]["chunks_indexed"]
                time_difference_seconds = round(
                    full_rebuild_baseline_seconds - elapsed_seconds,
                    3,
                )
            else:
                full_rebuild_baseline_seconds = None
                full_rebuild_baseline_chunks = None
                time_difference_seconds = None

            with st.expander("Update Efficiency Details", expanded=False):
                efficiency_columns = st.columns(2)

                with efficiency_columns[0]:
                    st.metric(
                        "Work Avoided",
                        f"{avoided_percent:.1f}%",
                        f"{avoided_chunks} chunks skipped",
                    )

                with efficiency_columns[1]:
                    if time_difference_seconds is None:
                        st.metric(
                            "Runtime Difference",
                            "No baseline",
                            "Run full rebuild first",
                        )
                    elif time_difference_seconds >= 0:
                        st.metric(
                            "Runtime Difference",
                            f"{time_difference_seconds}s faster",
                            f"vs {full_rebuild_baseline_seconds}s full rebuild",
                        )
                    else:
                        st.metric(
                            "Runtime Difference",
                            f"{abs(time_difference_seconds)}s slower",
                            f"vs {full_rebuild_baseline_seconds}s full rebuild",
                        )
                st.progress(refreshed_percent)

                st.caption(
                    f"Incremental update avoided re-embedding {avoided_chunks} of "
                    f"{before_active_vectors} previous active chunks. "
                    f"{chunks_refreshed} new chunks were embedded for the changed document(s)."
                )
                if latest_full_rebuild_result:
                    st.dataframe(
                        [
                            {
                                "Method": "Full Active Rebuild",
                                "Scope": "All active documents",
                                "Chunks Processed": full_rebuild_baseline_chunks,
                                "Elapsed Time": f"{full_rebuild_baseline_seconds}s",
                                "Use Case": "Clean full index reconstruction",
                            },
                            {
                                "Method": "Incremental Update",
                                "Scope": f"{changed_document_count} changed {document_label}",
                                "Chunks Processed": chunks_refreshed,
                                "Elapsed Time": f"{elapsed_seconds}s",
                                "Use Case": "Normal document update/sync",
                            },
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(
                        "Run a full rebuild benchmark first to compare incremental update "
                        "against a measured rebuild baseline."
                    )
        elif benchmark_type == "full_rebuild":
            elapsed_seconds = index_benchmark_results["elapsed_seconds"]
            chunks_indexed = index_benchmark_results["rebuild_result"]["chunks_indexed"]

            st.info(
                f"Latest benchmark was a full active-aware rebuild. "
                f"{chunks_indexed} active chunks were rebuilt in {elapsed_seconds}s."
            )

            metric_columns = st.columns(4)

            with metric_columns[0]:
                st.metric("Rebuild Time", f"{elapsed_seconds}s")

            with metric_columns[1]:
                st.metric("Chunks Rebuilt", chunks_indexed)

            with metric_columns[2]:
                st.metric("Active Records", active_records)

            with metric_columns[3]:
                 active_index_delta = (
                "Portal only"
                if db_size_mb is None
                else f"{db_size_mb} MB"
            )

            st.metric(
                "Active Index",
                f"{active_vectors} vectors",
                active_index_delta,
            )

        else:
            st.info("Latest result is an index snapshot.")

            metric_columns = st.columns(3)

            with metric_columns[0]:
                st.metric("Active Records", active_records)

            with metric_columns[1]:
                st.metric("Active Index", f"{active_vectors} vectors")

            with metric_columns[2]:
                if db_size_mb is None:
                    st.metric("Index Size", "Portal only")
                else:
                    st.metric("Index Size", f"{db_size_mb} MB")

        if archived_file_count:
            st.warning(
                f"{archived_file_count} archived source file(s) remain on disk for audit, "
                "but active-aware indexing excludes archived versions from the configured search index."
            )
        else:
            st.caption("Physical source files and active metadata records are aligned.")

        with st.expander("Technical index benchmark details", expanded=False):
            detail_rows = [
                {"Metric": "Benchmark Type", "Value": benchmark_type},
                {"Metric": "Active Metadata Records", "Value": active_records},
                {"Metric": "Physical Source Files", "Value": physical_files},
                {"Metric": "Archived Physical Files", "Value": archived_file_count},
                {"Metric": "Vector Backend", "Value": vector_backend},
                {"Metric": "Indexed Chunks", "Value": active_vectors},
                {"Metric": "Index Size MB", "Value": "Portal only" if db_size_mb is None else db_size_mb},
            ]

            if benchmark_type == "batch_incremental_update":
                detail_rows.extend(
                    [
                        {
                            "Metric": "Updated Sources",
                            "Value": ", ".join(index_benchmark_results["updated_sources"]),
                        },
                        {
                            "Metric": "Deleted Vectors",
                            "Value": index_benchmark_results["total_deleted_vectors"],
                        },
                        {
                            "Metric": "Chunks Re-indexed",
                            "Value": index_benchmark_results["total_chunks_indexed"],
                        },
                        {
                            "Metric": "Unchanged Chunks Avoided",
                            "Value": index_benchmark_results["estimated_unchanged_chunks_avoided"],
                        },
                        {
                            "Metric": "Elapsed Seconds",
                            "Value": index_benchmark_results["elapsed_seconds"],
                        },
                    ]
                )

            st.dataframe(
                detail_rows,
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.info(
            "No vector index benchmark found yet. Run "
            "`python -m src.evaluation.index_benchmark` to generate one."
        )

    st.divider()

    with st.container(border=True):
        st.subheader("Daily Average Query Latency")

        latency_rows = []

        for row in query_log_summary["daily_latency_rows"]:
            query_date = datetime.fromisoformat(row[0])

            latency_rows.append(
                {
                    "Date Key": row[0],
                    "Display Label": query_date.strftime("%a %d %b"),
                    "Day": query_date.strftime("%a"),
                    "Date": query_date.strftime("%d %b"),
                    "Average Latency (seconds)": round(row[2], 2),
                    "Query Count": row[1],
                }
            )

        if latency_rows:
            latency_data = pd.DataFrame(latency_rows).sort_values("Date Key")

            latency_chart = (
                alt.Chart(latency_data)
                .mark_bar(
                    color="#6f8bc7",
                    cornerRadiusTopLeft=3,
                    cornerRadiusTopRight=3,
                    size=80,
                )
                .encode(
                    x=alt.X(
                        "Display Label:N",
                        title="Day",
                        sort=None,
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y(
                        "Average Latency (seconds):Q",
                        title="Average Latency (seconds)",
                    ),
                    tooltip=[
                        alt.Tooltip("Day:N"),
                        alt.Tooltip("Date:N"),
                        alt.Tooltip("Average Latency (seconds):Q", format=".2f"),
                        alt.Tooltip("Query Count:Q"),
                    ],
                )
                .properties(height=280)
            )

            st.altair_chart(latency_chart, use_container_width=True)
        else:
            st.info("No query latency data yet. Submit a Chat query to populate this chart.")

    with st.expander("How benchmark accuracy is measured"):
        st.markdown(
            "**Eval Top-K Accuracy checks whether the expected source document appears "
            "within the top 5 retrieved chunks in the labelled evaluation run.**"
        )
        st.caption(
            "Top-K Accuracy and Miss Rate are calculated from the latest labelled "
            "retrieval evaluation result. Each labelled query defines an expected source "
            "or expected miss/block outcome."
        )
        if evaluation_results and evaluation_results.get("threshold_interpretation"):
            threshold_interpretation = evaluation_results["threshold_interpretation"]

            st.markdown("**Relevance Threshold Comparison**")
            st.caption(
                "This compares candidate retrieval score thresholds using the same "
                "labelled query set. It helps justify whether a lower threshold "
                "recovers useful documents or only adds noise."
            )

            st.dataframe(
                threshold_interpretation["comparison_rows"],
                use_container_width=True,
                hide_index=True,
            )

            if threshold_interpretation["has_clear_winner"]:
                st.success(threshold_interpretation["recommendation"])
            else:
                st.info(threshold_interpretation["recommendation"])

    with st.expander("Admin Review Queue - Live Issues & Benchmark Misses", expanded=False):
        st.markdown("**Live Query Issues / User Feedback**")
        live_review_rows = [
            {
                "Log ID": row[0],
                "Timestamp": row[1],
                "User": row[2],
                "Role": row[3],
                "Department": row[4],
                "Question": row[5],
                "Status": row[6],
                "Sources": ", ".join(json.loads(row[7])),
                "Latency (s)": row[8],
                "Feedback": row[9],
                "Note": row[10],
            }
            for row in query_log_summary["review_rows"]
        ]

        if live_review_rows:
            st.dataframe(live_review_rows, use_container_width=True, hide_index=True)
        else:
            st.success("No live query issues or reported feedback in the latest logged queries.")

        st.divider()
        st.markdown("**Labelled Benchmark Misses**")
        if evaluation_results and evaluation_results["miss_rows"]:
            real_miss_rows = [
                {
                    "Query ID": row["query_id"],
                    "Suite": row["suite"],
                    "Question": row["question"],
                    "Expected Source": row["expected_source"],
                    "Retrieved Sources": ", ".join(row["retrieved_sources"]),
                    "Issue": row["issue"],
                    "Next Enhancement": "Review metadata, chunking, filters, or Top-K ranking",
                }
                for row in evaluation_results["miss_rows"]
            ]

            st.dataframe(
                real_miss_rows,
                use_container_width=True,
                hide_index=True,
            )

        elif evaluation_results:
            st.success("No retrieval misses in the latest labelled retrieval evaluation.")

        else:
            st.info(
                "No retrieval evaluation result found yet. Run "
                "`python -m src.evaluation.retrieval_eval` to generate miss review data."
            )

    with st.expander(
        f"Query History - Latest {query_log_summary['query_history_limit']} Logged Queries",
        expanded=False,
    ):
        st.caption(
            "This is the persisted local chat/query history used for audit-style review, "
            "debugging, and dashboard signals."
        )
        recent_query_rows = [
            {
                "Timestamp": row[0],
                "User": row[1],
                "Role": row[2],
                "Department": row[3],
                "Question": row[4],
                "Department Filter": row[5],
                "File Type Filter": row[6],
                "Status": row[7],
                "Latency (s)": row[8],
            }
            for row in query_log_summary["recent_queries"]
        ]

        if recent_query_rows:
            st.dataframe(
                recent_query_rows,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No logged chat queries yet. Submit a Chat query to create a log.")


elif selected_page in ["KB Management", "KB Status"]:
    st.header(selected_page)

    if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
        st.caption("Global knowledge base management for all departments.")
    elif st.session_state["role"] == PROJECT_MANAGER_ROLE:
        st.caption("Department-scoped knowledge base management.")
    else:
        st.caption("Department-scoped knowledge base status.")

    documents = load_document_metadata()
    visible_documents = [
        document for document in documents
        if can_view_document(document)
    ]

    pending_review_documents = [
        document
        for document in load_document_metadata(include_inactive=True)
        if document.get("chunk_id") == "pending_review"
    ]

    summary_columns = st.columns(3)

    with summary_columns[0]:
        with st.container(border=True):
            st.markdown("**Source Connectors**")
            st.markdown(
                """
                <span style="
                    color: #1d4ed8;
                    background: #dbeafe;
                    border-radius: 0.3rem;
                    padding: 0.2rem 0.45rem;
                    font-size: 0.7rem;
                    font-weight: 700;
                ">DEMO CONNECTORS</span>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Integration direction:\n- SharePoint\n- OneNote\n- Manual Upload\n- Batch ZIP")

    with summary_columns[1]:
        with st.container(border=True):
            st.markdown("**Document Metadata & ACL**")
            st.markdown(
                """
                <span style="
                    color: #166534;
                    background: #dcfce7;
                    border-radius: 0.3rem;
                    padding: 0.2rem 0.45rem;
                    font-size: 0.7rem;
                    font-weight: 700;
                ">ACTIVE</span>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"Role: {escape(st.session_state['role'])}\n"
                f"- Visible docs: {len(visible_documents)}"
            )

    with summary_columns[2]:
        with st.container(border=True):
            st.markdown("**Visual Content Handling**")
            st.markdown(
                """
                <span style="
                    color: #92400e;
                    background: #fef3c7;
                    border-radius: 0.3rem;
                    padding: 0.2rem 0.45rem;
                    font-size: 0.7rem;
                    font-weight: 700;
                ">PARTIAL SUPPORT</span>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Active: Text extract\n"
                "Roadmap: OCR, Diagram extract"
            )
    
    if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:

        st.subheader("Document Ingestion & Indexing")

        if "upload_message" not in st.session_state:
            st.session_state["upload_message"] = ""

        if st.session_state["role"] in [SYSTEM_ADMIN_ROLE, PROJECT_MANAGER_ROLE]:
            with st.container(border=True):
                new_document_tab, new_version_tab = st.tabs(
                    ["Upload New Document", "Upload New Version"]
                )
                with new_document_tab:
                    st.markdown("**1. Upload & Categorize Document**")
                    st.caption(
                        "Upload captures document title, department, category, tags, and access scope. "
                        "File identity, storage location, version, uploader, timestamp, index status, and extraction status are assigned by the system."
                    )

                    if "upload_form_version" not in st.session_state:
                        st.session_state["upload_form_version"] = 0

                    upload_form_version = st.session_state["upload_form_version"]

                    uploaded_file = st.file_uploader(
                        "Upload TXT, PDF, or DOCX",
                        type=["txt", "pdf", "docx"],
                        key=f"upload_file{upload_form_version}",
                    )

                    title_key = prepare_upload_title_state(uploaded_file, upload_form_version)

                    with st.form(f"real_txt_upload_form_{upload_form_version}"):
                        
                        # Grouping form inputs into a grid layout to save vertical space
                        col1, col2 = st.columns(2)

                        with col1:
                            title = st.text_input(
                                "Document title",
                                key=title_key,
                                help="Auto-filled from the uploaded filename. Admin may edit it.",
                            )
                            if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
                                department = st.selectbox(
                                    "Department",
                                    DEPARTMENT_OPTIONS,
                                    key=f"txt_upload_department_{upload_form_version}",
                                )
                            else:
                                department = st.text_input(
                                    "Department",
                                    value=st.session_state["department"],
                                    disabled=True,
                                    key=f"txt_upload_department_{upload_form_version}",
                                )
                            category = st.text_input(
                                "Category",
                                value="General",
                                key=f"txt_upload_category_{upload_form_version}",
                            )

                        with col2:
                            tags_text = st.text_input(
                                "Tags",
                                value="policy, internal",
                                help="Separate tags with commas.",
                                key=f"txt_upload_tags_{upload_form_version}",
                            )
                            if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
                                allowed_roles = st.multiselect(
                                    "Allowed roles",
                                    ROLE_OPTIONS,
                                    default=[SYSTEM_ADMIN_ROLE],
                                    key=f"txt_upload_roles_{upload_form_version}",
                                )
                                allowed_departments = st.multiselect(
                                    "Allowed departments",
                                    DEPARTMENT_OPTIONS,
                                    default=[department],
                                    key=f"txt_upload_departments_{upload_form_version}",
                                )
                            else:
                                allowed_roles = st.multiselect(
                                    "Allowed roles",
                                    [PROJECT_MANAGER_ROLE, GENERAL_EMPLOYEE_ROLE],
                                    default=[PROJECT_MANAGER_ROLE],
                                    key=f"txt_upload_roles_{upload_form_version}",
                                )
                                allowed_departments = st.multiselect(
                                    "Allowed departments",
                                    [st.session_state["department"]],
                                    default=[st.session_state["department"]],
                                    key=f"txt_upload_departments_{upload_form_version}",
                                )

                        submitted_upload = st.form_submit_button("Save File + Metadata", type="primary")

                        if submitted_upload:
                            if uploaded_file is None:
                                st.error("Please choose a supported file before saving.")
                            elif not title.strip():
                                st.error("Please enter a document title.")
                            elif not allowed_roles:
                                st.error("Please select at least one allowed role.")
                            elif not allowed_departments:
                                st.error("Please select at least one allowed department.")
                            else:
                                tags = [
                                    tag.strip()
                                    for tag in tags_text.split(",")
                                    if tag.strip()
                                ]

                                try:
                                    upload_result = request_backend_document_upload(
                                        uploaded_file=uploaded_file,
                                        title=title.strip(),
                                        document_department=department,
                                        category=category.strip() or "General",
                                        tags=tags,
                                        allowed_roles=allowed_roles,
                                        allowed_departments=allowed_departments,
                                    )
                                except requests.exceptions.HTTPError as error:
                                    st.error(f"Upload rejected by backend: {error.response.text}")
                                    st.stop()
                                except requests.exceptions.RequestException as error:
                                    st.error(f"Could not upload document through backend: {error}")
                                    st.stop()

                                index_owner_message = (
                                    "Run Update for Pending Documents so the latest approved content is available in chat."
                                    if st.session_state["role"] == SYSTEM_ADMIN_ROLE
                                    else "System Admin action is required to update the search index before this content is available in chat."
                                )

                                st.session_state["upload_message"] = (
                                    f"{upload_result['message']} {index_owner_message}"
                                )
                                st.session_state["upload_form_version"] += 1
                                st.rerun()

                with new_version_tab:
                    st.markdown("**Upload Replacement As New Version**")
                    st.caption(
                        "Select an active document and upload its replacement. "
                        "The system archives the previous version, creates the next version record, and marks it for search index update."
                    )

                    manageable_documents = [
                        document for document in visible_documents
                        if (
                            st.session_state["role"] == SYSTEM_ADMIN_ROLE
                            or document["department"] == st.session_state["department"]
                        )
                    ]

                    if not manageable_documents:
                        st.info("No manageable documents are available for version replacement.")
                    else:
                        document_options = {
                            f"{document['title']} ({document['document_id']}, v{document.get('version_number') or 1})": document
                            for document in manageable_documents
                        }

                        selected_version_label = st.selectbox(
                            "Existing document",
                            list(document_options.keys()),
                            key=f"version_replace_document_{upload_form_version}",
                        )

                        selected_version_document = document_options[selected_version_label]
                        previous_version_number = selected_version_document.get("version_number") or 1
                        next_version_number = previous_version_number + 1

                        st.caption(
                            f"Current version: v{previous_version_number} -> New version: v{next_version_number}"
                        )

                        uploaded_version_file = st.file_uploader(
                            "Upload replacement TXT, PDF, or DOCX",
                            type=["txt", "pdf", "docx"],
                            key=f"version_upload_file_{upload_form_version}",
                        )

                        with st.form(f"version_upload_form_{upload_form_version}"):
                            submitted_version_upload = st.form_submit_button(
                                "Create New Version",
                                type="primary",
                            )

                            if submitted_version_upload:
                                if uploaded_version_file is None:
                                    st.error("Please choose a replacement file before saving.")
                                else:
                                    try:
                                        version_result = request_backend_document_version_upload(
                                            uploaded_file=uploaded_version_file,
                                            previous_document_id=selected_version_document["document_id"],
                                        )
                                    except requests.exceptions.HTTPError as error:
                                        st.error(f"Version upload rejected by backend: {error.response.text}")
                                        st.stop()
                                    except requests.exceptions.RequestException as error:
                                        st.error(f"Could not upload replacement version through backend: {error}")
                                        st.stop()

                                    index_owner_message = (
                                        "Run Update for Pending Documents to replace old search vectors and activate the new version in chat."
                                        if st.session_state["role"] == SYSTEM_ADMIN_ROLE
                                        else "System Admin action is required to replace old search vectors and activate the new version in chat."
                                    )

                                    st.session_state["upload_message"] = (
                                        f"{version_result['message']} {index_owner_message}"
                                    )
                                    st.session_state["upload_form_version"] += 1
                                    st.rerun()

        if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
            with st.container(border=True):
                st.markdown("**2. Vector Index Sync**")
                st.caption(
                    "Use incremental update for pending document changes. Use full rebuild "
                    "when you want to reconstruct the active index from scratch."
                )

                if st.session_state.get("index_update_job_message"):
                    index_update_status = st.session_state.get("index_update_job_status", "info")

                    if index_update_status == "success":
                        st.success(st.session_state["index_update_job_message"])
                    elif index_update_status == "error":
                        st.error(st.session_state["index_update_job_message"])
                    else:
                        st.info(st.session_state["index_update_job_message"])

                if st.session_state.get("reindex_job_message"):
                    reindex_status = st.session_state.get("reindex_job_status", "info")

                    if reindex_status == "success":
                        st.success(st.session_state["reindex_job_message"])
                    elif reindex_status == "error":
                        st.error(st.session_state["reindex_job_message"])
                    else:
                        st.info(st.session_state["reindex_job_message"])

                index_action_columns = st.columns(2)

                with index_action_columns[0]:
                    if st.button(
                        "Run Update for Pending Documents",
                        use_container_width=True,
                        disabled=bool(st.session_state.get("active_index_update_job_id")),
                    ):
                        try:
                            job = submit_index_update_job()
                        except requests.exceptions.RequestException as error:
                            st.error(f"Could not submit index update job: {error}")
                        else:
                            st.session_state["active_index_update_job_id"] = job["job_id"]
                            st.session_state["index_update_job_message"] = "Pending document index update queued."
                            st.session_state["index_update_job_status"] = "info"
                            st.rerun()

                with index_action_columns[1]:
                   if st.button(
                        "Rebuild Full Active Index",
                        use_container_width=True,
                        disabled=bool(st.session_state.get("active_reindex_job_id")),
                    ):
                        try:
                            job = submit_reindex_job()
                        except requests.exceptions.RequestException as error:
                            st.error(f"Could not submit rebuild job: {error}")
                        else:
                            st.session_state["active_reindex_job_id"] = job["job_id"]
                            st.session_state["reindex_job_message"] = "Search index rebuild queued."
                            st.session_state["reindex_job_status"] = "info"
                            st.rerun()

        if st.session_state["upload_message"]:
            st.warning(st.session_state["upload_message"])

    st.divider()

    if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
        st.subheader("OneDrive Connector")

        with st.container(border=True):
            st.caption(
                "Scans the configured OneDrive knowledge-base root only. "
                "Selected files are staged for metadata and ACL review before indexing."
            )

            if st.button("Scan OneDrive Root", use_container_width=True):
                try:
                    scan_result = request_onedrive_file_scan()
                except requests.exceptions.HTTPError as error:
                    st.error(f"OneDrive scan rejected by backend: {error.response.text}")
                except requests.exceptions.RequestException as error:
                    st.error(f"Could not scan OneDrive connector: {error}")
                else:
                    st.session_state["onedrive_files"] = scan_result["files"]
                    st.success(f"Found {len(scan_result['files'])} file(s).")

            onedrive_files = st.session_state.get("onedrive_files", [])

            if onedrive_files:
                st.dataframe(
                    [
                        {
                            "Name": file_item["name"],
                            "Path": file_item["connector_path"],
                            "Size": file_item.get("size"),
                            "Modified": file_item.get("last_modified_datetime"),
                        }
                        for file_item in onedrive_files
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                file_options = {
                    f"{file_item['name']} - {file_item['connector_path']}": file_item
                    for file_item in onedrive_files
                }

                selected_file_label = st.selectbox(
                    "Select OneDrive file to stage",
                    list(file_options.keys()),
                    key="selected_onedrive_file_to_stage",
                )

                if st.button("Stage Selected File for Review", use_container_width=True):
                    try:
                        stage_result = request_onedrive_file_stage(
                            file_options[selected_file_label]
                        )
                    except requests.exceptions.HTTPError as error:
                        st.error(f"OneDrive staging rejected by backend: {error.response.text}")
                    except requests.exceptions.RequestException as error:
                        st.error(f"Could not stage OneDrive file: {error}")
                    else:
                        st.success(stage_result["message"])
                        st.rerun()

        st.divider()

    if st.session_state["role"] in ["System Admin", "Project Manager"]:
        st.subheader("Pending Connector Review")

        reviewable_documents = [
            document
            for document in pending_review_documents
            if (
                st.session_state["role"] == "System Admin"
                or document["department"] == st.session_state["department"]
            )
        ]

        if not reviewable_documents:
            st.caption("No connector documents are waiting for your review.")
        else:
            review_rows = [
                {
                    "Document ID": document["document_id"],
                    "Title": document["title"],
                    "Source": document["source"],
                    "Suggested Department": document["department"],
                    "Storage URI": document.get("storage_uri", ""),
                    "Uploaded At": document.get("uploaded_at", ""),
                }
                for document in reviewable_documents
            ]

            st.dataframe(
                review_rows,
                use_container_width=True,
                hide_index=True,
            )

            document_options = {
                f"{document['title']} ({document['document_id']})": document
                for document in reviewable_documents
            }

            selected_review_label = st.selectbox(
                "Select connector document to review",
                list(document_options.keys()),
                key="selected_connector_review_document",
            )

            selected_review_document = document_options[selected_review_label]

            with st.form(f"approve_connector_{selected_review_document['document_id']}"):
                edited_title = st.text_input(
                    "Title",
                    value=selected_review_document["title"],
                    key=f"approve_title_{selected_review_document['document_id']}",
                )

                edited_department = st.selectbox(
                    "Department",
                    DEPARTMENT_OPTIONS,
                    index=DEPARTMENT_OPTIONS.index(selected_review_document["department"])
                    if selected_review_document["department"] in DEPARTMENT_OPTIONS
                    else 0,
                    key=f"approve_department_{selected_review_document['document_id']}",
                )

                edited_category = st.text_input(
                    "Category",
                    value=selected_review_document["category"],
                    key=f"approve_category_{selected_review_document['document_id']}",
                )

                edited_tags_text = st.text_input(
                    "Tags",
                    value=", ".join(selected_review_document["tags"]),
                    key=f"approve_tags_{selected_review_document['document_id']}",
                )

                edited_allowed_roles = st.multiselect(
                    "Allowed roles",
                    ROLE_OPTIONS,
                    default=[
                        role
                        for role in selected_review_document["allowed_roles"]
                        if role in ROLE_OPTIONS
                    ],
                    key=f"approve_roles_{selected_review_document['document_id']}",
                )

                edited_allowed_departments = st.multiselect(
                    "Allowed departments",
                    DEPARTMENT_OPTIONS,
                    default=[
                        department
                        for department in selected_review_document["allowed_departments"]
                        if department in DEPARTMENT_OPTIONS
                    ],
                    key=f"approve_departments_{selected_review_document['document_id']}",
                )

                submitted_approval = st.form_submit_button("Approve for Indexing")

                if submitted_approval:
                    try:
                        approval_result = request_document_approval(
                            document_id=selected_review_document["document_id"],
                            title=edited_title,
                            department=edited_department,
                            category=edited_category,
                            tags=[
                                tag.strip()
                                for tag in edited_tags_text.split(",")
                                if tag.strip()
                            ],
                            allowed_roles=edited_allowed_roles,
                            allowed_departments=edited_allowed_departments,
                        )
                    except requests.exceptions.HTTPError as error:
                        st.error(f"Approval rejected by backend: {error.response.text}")
                    except requests.exceptions.RequestException as error:
                        st.error(f"Could not approve connector document: {error}")
                    else:
                        st.success(approval_result["message"])
                        st.rerun()

    st.divider()
    if not visible_documents:
        st.warning("No documents are visible for the current role and department.")
    else:
        department_options = sorted(
            {document["department"] for document in visible_documents}
        )
        category_options = sorted(
            {document["category"] for document in visible_documents}
        )
        source_options = sorted(
            {document["source"] for document in visible_documents}
        )

        index_status_options = [
            FILTER_ALL,
            "Pending Index",
            "Indexed",
        ]

        filtered_documents = visible_documents

        if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
            indexed_count = sum(
                1 for document in visible_documents
                if get_index_status_label(document) == "Indexed"
            )

            pending_index_count = sum(
                1 for document in visible_documents
                if get_index_status_label(document) == "Pending Index"
            )

            versioned_count = sum(
                1 for document in visible_documents
                if (document.get("version_number") or 1) > 1
            )

            with st.container(border=True):
                st.markdown("**Index & Version Overview**")
                st.caption(
                    "Operational view for documents that may need incremental indexing "
                    "after upload, metadata update, or version replacement."
                )

                if pending_index_count:
                    if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
                        st.warning(
                            f"{pending_index_count} document(s) require search index update before chat can use the latest content. "
                            "Use Run Update for Pending Documents in Document Ingestion & Indexing."
                        )
                    else:
                        st.warning(
                            f"{pending_index_count} document(s) are waiting for System Admin search index update before chat can use the latest content."
                        )
                else:
                    st.success("All visible active documents are indexed.")

                status_columns = st.columns(4)

                with status_columns[0]:
                    st.metric("Visible Documents", len(visible_documents))

                with status_columns[1]:
                    st.metric(
                        "Pending Index",
                        pending_index_count,
                        "Action required" if pending_index_count else "No action needed",
                    )

                with status_columns[2]:
                    st.metric("Indexed", indexed_count)

                with status_columns[3]:
                    st.metric("Updated Versions", versioned_count)

        if st.session_state["role"] == GENERAL_EMPLOYEE_ROLE:
            st.subheader("Available Knowledge Base Documents")
            st.caption(
                "Documents shown here are limited to your department and ACL-permitted shared sources."
            )
        else:
            st.subheader("Document Index & Filter")
            
        with st.container(border=True):
            filter_columns = st.columns(4 if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE else 3)

            with filter_columns[0]:
                selected_department = st.selectbox(
                    "Visible Document Department",
                    [FILTER_ALL] + department_options,
                )

            with filter_columns[1]:
                selected_category = st.selectbox(
                    "Category",
                    [FILTER_ALL] + category_options,
                )

            with filter_columns[2]:
                selected_source = st.selectbox(
                    "Source",
                    [FILTER_ALL] + source_options,
                )

            selected_index_status = FILTER_ALL

            if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
                with filter_columns[3]:
                    selected_index_status = st.selectbox(
                        "Index Status",
                        index_status_options,
                    )

        if selected_department != FILTER_ALL:
            filtered_documents = [
                document for document in filtered_documents
                if document["department"] == selected_department
            ]

        if selected_category != FILTER_ALL:
            filtered_documents = [
                document for document in filtered_documents
                if document["category"] == selected_category
            ]

        if selected_source != FILTER_ALL:
            filtered_documents = [
                document for document in filtered_documents
                if document["source"] == selected_source
            ]

        if selected_index_status != FILTER_ALL:
            filtered_documents = [
                document
                for document in filtered_documents
                if get_index_status_label(document) == selected_index_status
            ]

        table_rows = []

        for document in filtered_documents:
            if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
                row = {
                    "Document": document["title"],
                    "Version": get_version_label(document),
                    "Index Status": get_index_status_label(document),
                    "Source": document["source"],
                    "Storage": document.get("storage_backend", "local"),
                    "Department": document["department"],
                    "Category": document["category"],
                    "Allowed Access": ", ".join(document["allowed_roles"]),
                    "Visuals": document["visual_extraction_status"],
                }
            else:
                row = {
                    "Document": document["title"],
                    "Source": document["source"],
                    "Storage": document.get("storage_backend", "local"),
                    "Department": document["department"],
                    "Category": document["category"],
                    "Visuals": document["visual_extraction_status"],
                }

            table_rows.append(row)

        st.dataframe(
            table_rows,
            use_container_width=True,
            hide_index=True,
        )

        if filtered_documents:
            st.subheader("Selected Document Details")
            selected_title = st.selectbox(
                "Select document to view details",
                [document["title"] for document in filtered_documents],
            )

            selected_document = next(
                document for document in filtered_documents
                if document["title"] == selected_title
            )

            detail_columns = st.columns(2)

            with detail_columns[0]:
                with st.container(border=True):
                    st.markdown("**File Metadata**")
                    st.write(f"**ID:** {selected_document['document_id']}")
                    st.write(f"**File:** {selected_document['filename']} | **Type:** {selected_document['file_type']}")
                    st.write(f"**Storage backend:** {selected_document.get('storage_backend', 'local')}")
                    st.code(selected_document.get("storage_uri", f"data/simulated/{selected_document['filename']}"), language=None)
                    st.write(f"**Uploaded by:** {selected_document['uploaded_by']} at {selected_document['uploaded_at']}")
                    if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
                        st.write(f"**Source document ID:** {selected_document.get('source_document_id')}")
                        st.write(f"**Version:** {get_version_label(selected_document)}")
                        st.write(f"**Status:** {get_index_status_label(selected_document)}")

            with detail_columns[1]:
                with st.container(border=True):
                    st.markdown("**Access & Extraction**")
                    st.write(f"**Tags:** {', '.join(selected_document['tags'])}")
                    if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
                        st.write(
                            "**Allowed roles:** "
                            f"{', '.join(selected_document['allowed_roles'])}"
                        )
                        st.write(
                            "**Allowed departments:** "
                            f"{', '.join(selected_document['allowed_departments'])}"
                        )
                        st.write(f"**Index marker:** {selected_document['chunk_id']}")
                    st.write(
                        "**Visual extraction status:** "
                        f"{selected_document['visual_extraction_status']}"
                    )

            if st.session_state["role"] in [SYSTEM_ADMIN_ROLE, PROJECT_MANAGER_ROLE]:
                with st.expander("Archive Document", expanded=False):
                    st.warning(
                        "Archiving removes this document from active retrieval and deletes its vector/index records from the configured backend. "
                        "Use this for retired, duplicate, or outdated documents without a replacement."
                    )

                    can_archive_selected_document = (
                        st.session_state["role"] == SYSTEM_ADMIN_ROLE
                        or selected_document["department"] == st.session_state["department"]
                    )

                    if not can_archive_selected_document:
                        st.info("Project Manager can only archive own-department documents.")
                    else:
                        confirm_archive = st.checkbox(
                            f"I understand this will archive {selected_document['title']}.",
                            key=f"confirm_archive_{selected_document['document_id']}",
                        )

                        if st.button(
                            "Archive Selected Document",
                            key=f"archive_document_{selected_document['document_id']}",
                            disabled=not confirm_archive,
                        ):
                            try:
                                archive_result = request_document_archive(
                                    selected_document["document_id"]
                                )
                            except requests.exceptions.HTTPError as error:
                                st.error(f"Archive rejected by backend: {error.response.text}")
                            except requests.exceptions.RequestException as error:
                                st.error(f"Could not archive document: {error}")
                            else:
                                st.success(archive_result["message"])
                                st.rerun()
                with st.expander("Edit Metadata & Access", expanded=False):
                    with st.form(f"metadata_edit_form_{selected_document['document_id']}"):
                        edit_col1, edit_col2 = st.columns(2)
                        
                        with edit_col1:
                            edited_title = st.text_input(
                                "Title",
                                value=selected_document["title"],
                            )

                            if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
                                edited_department = st.selectbox(
                                    "Department",
                                    DEPARTMENT_OPTIONS,
                                    index=DEPARTMENT_OPTIONS.index(selected_document["department"]),
                                )
                            else:
                                edited_department = st.text_input(
                                    "Department",
                                    value=st.session_state["department"],
                                    disabled=True,
                                )

                            edited_category = st.text_input(
                                "Category",
                                value=selected_document["category"],
                            )

                        with edit_col2:
                            edited_tags_text = st.text_input(
                                "Tags",
                                value=", ".join(selected_document["tags"]),
                                help="Separate tags with commas.",
                            )

                            if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
                                edited_allowed_roles = st.multiselect(
                                    "Allowed roles",
                                    ROLE_OPTIONS,
                                    default=selected_document["allowed_roles"],
                                )
                                edited_allowed_departments = st.multiselect(
                                    "Allowed departments",
                                    [FILTER_ALL] + DEPARTMENT_OPTIONS,
                                    default=selected_document["allowed_departments"],
                                )
                            else:
                                edited_allowed_roles = st.multiselect(
                                    "Allowed roles",
                                    [PROJECT_MANAGER_ROLE, GENERAL_EMPLOYEE_ROLE],
                                    default=[
                                        role for role in selected_document["allowed_roles"]
                                        if role in [PROJECT_MANAGER_ROLE, GENERAL_EMPLOYEE_ROLE]
                                    ] or [PROJECT_MANAGER_ROLE],
                                )
                                st.text_input(
                                    "Allowed departments",
                                    value=st.session_state["department"],
                                    disabled=True,
                                    help="Project Manager metadata edits are limited to their own department.",
                                )
                                edited_allowed_departments = [st.session_state["department"]]

                        submitted_metadata_update = st.form_submit_button("Save Metadata", type="primary")

                        if submitted_metadata_update:
                            if not edited_title.strip():
                                st.error("Please enter a document title.")
                            elif not edited_allowed_roles:
                                st.error("Please select at least one allowed role.")
                            elif not edited_allowed_departments:
                                st.error("Please select at least one allowed department.")
                            else:
                                try:
                                    approved_metadata = request_metadata_update_validation(
                                        document_department=edited_department,
                                        allowed_roles=edited_allowed_roles,
                                        allowed_departments=edited_allowed_departments,
                                    )
                                except requests.exceptions.HTTPError as error:
                                    st.error(f"Metadata update rejected by backend: {error.response.text}")
                                except requests.exceptions.RequestException as error:
                                    st.error(f"Could not validate metadata update: {error}")
                                else:
                                    updated_document = selected_document.copy()
                                    updated_document.update(
                                        {
                                            "title": edited_title.strip(),
                                            "department": approved_metadata["document_department"],
                                            "category": edited_category.strip() or "General",
                                            "tags": [
                                                tag.strip()
                                                for tag in edited_tags_text.split(",")
                                                if tag.strip()
                                            ],
                                            "allowed_roles": approved_metadata["allowed_roles"],
                                            "allowed_departments": approved_metadata["allowed_departments"],
                                        }
                                    )

                                    update_document_metadata(
                                        selected_document["document_id"],
                                        updated_document,
                                    )

                                    st.success("Metadata updated. ACL changes apply to chat immediately.")
                                    st.rerun()
                    

elif selected_page == "Chat":
    st.header("Copilot Chat")
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    documents = load_document_metadata()
    visible_documents = [
        document for document in documents
        if can_view_document(document)
    ]

    available_departments = sorted(
        {document["department"] for document in visible_documents}
    )
    available_file_types = sorted(
        {document["file_type"] for document in visible_documents}
    )

    st.caption("Ask questions grounded in the enterprise knowledge base.")

    with st.expander("Search Scope & Filters", expanded=False):
        filter_columns = st.columns([1, 1, 2])
        if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
            with filter_columns[0]:
                department_filter = st.selectbox(
                    "Department",
                    [FILTER_ALL] + available_departments,
                )

            with filter_columns[1]:
                file_type_filter = st.selectbox(
                    "File Type",
                    [FILTER_ALL] + available_file_types,
                )

            filter_status = (
                f"Selected filter: Department = {department_filter}, "
                f"File Type = {file_type_filter}"
            )

        elif st.session_state["role"] == PROJECT_MANAGER_ROLE:
            department_filter = None

            with filter_columns[0]:
                st.text_input(
                    "Department",
                    value=(
                        f"{st.session_state['department']} + "
                        "ACL shared docs"
                    ),
                    disabled=True,
                )

            with filter_columns[1]:
                file_type_filter = st.selectbox(
                    "File Type",
                    [FILTER_ALL] + available_file_types,
                )

            filter_status = (
                f"Selected filter: Department = {st.session_state['department']} + "
                f"ACL-permitted shared documents, File Type = {file_type_filter}"
            )

        else:
            department_filter = None
            file_type_filter = FILTER_ALL

            with filter_columns[0]:
                st.text_input(
                    "Department",
                    value=st.session_state["department"],
                    disabled=True,
                )

            with filter_columns[1]:
                st.text_input(
                    "File Type",
                    value="Not available",
                    disabled=True,
                )

            filter_status = (
                "Scope: own department plus shared documents allowed by ACL. "
                "Advanced filters are not available for General Employee role."
            )

        with filter_columns[2]:
            st.text_input(
                "Selected Search Scope",
                value=filter_status,
                disabled=True,
            )

    chat_container = st.container(height=450, border=True)

    with chat_container:
        if not st.session_state["chat_messages"]:
            st.markdown(
                f"<div style='text-align: center; color: #94a3b8; margin-top: 2rem;'>"
                f"Hello {st.session_state['user']}, how can I help you today?"
                f"</div>",
                unsafe_allow_html=True,
            )

        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and message.get("status"):
                    show_status_message(message["status"])
                    if st.session_state["role"] != SYSTEM_ADMIN_ROLE:
                        show_escalation_guidance(message["status"])

                st.write(message["content"])

                # Wrap sources and context into columns and expanders to declutter the chat flow
                if message.get("sources") or message.get("context"):
                    meta_col1, meta_col2 = st.columns(2)
                    if message.get("sources"):
                        with meta_col1:
                            with st.expander("📑 View Sources"):
                                for source in message["sources"]:
                                    st.code(source, language=None)
                    
                    if message.get("context"):
                        with meta_col2:
                            with st.expander("🔍 Query Context"):
                                st.caption(message["context"])
                if (
                    message["role"] == "assistant"
                    and message.get("query_log_id")
                    and message.get("feedback", "none") == "none"
                ):
                    feedback_columns = st.columns(2)

                    with feedback_columns[0]:
                        if st.button(
                            "Helpful",
                            key=f"helpful_{message['query_log_id']}",
                            use_container_width=True,
                        ):
                            update_query_feedback(message["query_log_id"], "helpful")
                            message["feedback"] = "helpful"
                            st.rerun()

                    with feedback_columns[1]:
                        if st.button(
                            "Report Issue",
                            key=f"issue_{message['query_log_id']}",
                            use_container_width=True,
                        ):
                            update_query_feedback(message["query_log_id"], "reported_issue")
                            message["feedback"] = "reported_issue"
                            st.rerun()

                elif message["role"] == "assistant" and message.get("feedback") == "helpful":
                    st.caption("Feedback recorded: helpful")

                elif message["role"] == "assistant" and message.get("feedback") == "reported_issue":
                    st.caption("Feedback recorded: reported issue")

    chat_is_processing = st.session_state.get("chat_is_processing", False)

    with st.form("chat_question_form", border=False):
        question_columns = st.columns([6, 1])

        with question_columns[0]:
            st.text_input(
                "Message",
                key="chat_question",
                placeholder="Type or select an example question...",
                label_visibility="collapsed",
            )

        with question_columns[1]:
            st.form_submit_button(
                "Send",
                type="primary",
                use_container_width=True,
                on_click=submit_chat_question,
                disabled=chat_is_processing,
            )

    # Prompt buttons directly below the input
    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    example_prompts = ROLE_AWARE_CHAT_PROMPTS[st.session_state["role"]]
    
    # Calculate columns (Add +1 for Demo button if General Employee, or Clear button)
    total_cols = len(example_prompts) + (2 if st.session_state["role"] == GENERAL_EMPLOYEE_ROLE else 1)
    example_columns = st.columns(total_cols)

    for i, (label, prompt) in enumerate(example_prompts.items()):
        with example_columns[i]:
            st.button(
                label,
                key=f"example_prompt_{label}",
                on_click=select_example_chat_prompt,
                args=(prompt,),
                use_container_width=True,
                disabled=chat_is_processing,
            )

    if st.session_state["role"] == GENERAL_EMPLOYEE_ROLE:
        with example_columns[-2]:
            st.button(
                "Demo Restricted IT Policy",
                key="acl_demo_prompt",
                on_click=select_example_chat_prompt,
                args=("What are the password policy requirements?",),
                use_container_width=True,
                disabled=chat_is_processing,
            )

    with example_columns[-1]:
        if st.button("🧹 Clear", use_container_width=True, disabled=chat_is_processing):
            st.session_state["chat_messages"] = []
            st.rerun()

    question = st.session_state.pop("pending_chat_question", None)

    if question:
        clean_question = question.strip()

        if not clean_question:
            st.session_state["chat_is_processing"] = False
        else:
            user_message = {
                "role": "user",
                "content": clean_question,
            }
            st.session_state["chat_messages"].append(user_message)

            try:
                job = submit_chat_job(
                    clean_question,
                    department_filter,
                    file_type_filter,
                )
            except requests.exceptions.HTTPError as error:
                detail = "The question could not be submitted."

                try:
                    detail = error.response.json().get("detail", detail)
                except ValueError:
                    pass

                st.session_state["chat_messages"].append(
                    {
                        "role": "assistant",
                        "content": detail,
                        "sources": [],
                        "context": "",
                        "status": "not_found",
                    }
                )
                st.session_state["chat_is_processing"] = False
            except requests.exceptions.RequestException as error:
                st.session_state["chat_messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Could not submit chat job: {error}",
                        "sources": [],
                        "context": "",
                        "status": "connection_error",
                    }
                )
                st.session_state["chat_is_processing"] = False
            else:
                st.session_state["active_chat_job_id"] = job["job_id"]
                st.session_state["chat_is_processing"] = True

            st.rerun()

    poll_active_chat_job()

elif selected_page == "Settings":
    st.header("System Settings")
    st.caption("Admin-only backend mode, retrieval, and guardrail configuration.")

    try:
        settings_response = request_admin_settings()
        current_settings = settings_response["settings"]
    except requests.exceptions.RequestException as error:
        st.error(f"Could not load backend settings: {error}")
        st.stop()

    if st.session_state.get("settings_message"):
        if st.session_state.get("settings_rebuild_required"):
            st.warning(st.session_state["settings_message"])
        else:
            st.success(st.session_state["settings_message"])

    with st.container(border=True):
        st.subheader("Current Architecture")
        mode_columns = st.columns(5)
        mode_columns[0].metric("Storage", current_settings["storage_backend"])
        mode_columns[1].metric("Vector", current_settings["vector_backend"])
        mode_columns[2].metric("Embedding", current_settings["embedding_backend"])
        mode_columns[3].metric("LLM", current_settings["llm_backend"])
        mode_columns[4].metric("SharePoint", current_settings.get("sharepoint_mode", "simulated"))

    with st.form("runtime_settings_form", border=True):
        st.subheader("Configure Backend")
        
        st.markdown("**Infrastructure Backends**")
        infra_col1, infra_col2 = st.columns(2)
        
        with infra_col1:
            storage_backend = st.selectbox(
                "Storage Backend",
                options=["local", "azure_blob"],
                index=["local", "azure_blob"].index(current_settings["storage_backend"]),
                help="Determines where physical documents and files are stored."
            )
            embedding_backend = st.selectbox(
                "Embedding Backend",
                options=["local", "azure_openai"],
                index=["local", "azure_openai"].index(current_settings["embedding_backend"]),
                help="Model used to generate vector embeddings for chunks."
            )

        with infra_col2:
            vector_backend = st.selectbox(
                "Vector Backend",
                options=["chroma", "azure_search"],
                index=["chroma", "azure_search"].index(current_settings["vector_backend"]),
                help="Database used to store and query vector embeddings."
            )
            llm_backend = st.selectbox(
                "LLM Backend",
                options=["ollama", "azure_openai"],
                index=["ollama", "azure_openai"].index(current_settings["llm_backend"]),
                help="Primary text generation model."
            )

        st.divider()

        st.markdown("**Retrieval & Guardrails**")
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            top_k = st.number_input(
                "Top-K Results",
                min_value=1,
                max_value=20,
                value=int(current_settings["top_k"]),
                step=1,
                help="Number of document chunks retrieved per query."
            )
            
        with param_col2:
            minimum_relevance_threshold = st.number_input(
                "Min Relevance Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(current_settings["minimum_relevance_threshold"]),
                step=0.01,
                help="Filters out chunks with similarity scores below this threshold."
            )

        guardrail_prompt = st.text_area(
            "Admin Guardrail Prompt",
            value=current_settings["guardrail_prompt"],
            height=120,
            help="System prompt injected to enforce organizational rules on the LLM."
        )

        submitted_settings = st.form_submit_button("Save Runtime Settings", type="primary")

        if submitted_settings:
            risky_change = (
                vector_backend != current_settings["vector_backend"]
                or embedding_backend != current_settings["embedding_backend"]
            )

            if risky_change:
                st.warning(
                    "Changing vector or embedding backend requires rebuilding the active search index. "
                    "Until rebuild completes, queries continue using the previously built index."
                )

            try:
                update_response = request_settings_update({
                    "storage_backend": storage_backend,
                    "vector_backend": vector_backend,
                    "embedding_backend": embedding_backend,
                    "llm_backend": llm_backend,
                    "top_k": str(top_k),
                    "minimum_relevance_threshold": str(minimum_relevance_threshold),
                    "guardrail_prompt": guardrail_prompt,
                })
            except requests.exceptions.HTTPError as error:
                st.error(f"Settings rejected by backend: {error.response.text}")
            except requests.exceptions.RequestException as error:
                st.error(f"Could not save backend settings: {error}")
            else:
                st.session_state["settings_message"] = update_response["message"]
                st.session_state["settings_rebuild_required"] = update_response["rebuild_required"]
                st.session_state["settings_changed_keys"] = update_response.get("changed_keys", [])
                st.rerun()

    if st.session_state.get("settings_rebuild_required"):
        with st.container(border=True):
            st.subheader("Search Index Rebuild Required")
            st.info(
                "A vector or embedding mode change was detected. Run a full rebuild when ready "
                "to activate the selected retrieval configuration."
            )

            if st.button("Run Rebuild Now", type="primary", use_container_width=True):
                try:
                    job = submit_reindex_job()
                except requests.exceptions.RequestException as error:
                    st.session_state["settings_message"] = f"Could not submit rebuild job: {error}"
                    st.session_state["settings_rebuild_required"] = True
                    st.error(st.session_state["settings_message"])
                else:
                    st.session_state["active_reindex_job_id"] = job["job_id"]
                    st.session_state["settings_message"] = "Search index rebuild queued."
                    st.session_state["settings_rebuild_required"] = True
                    st.rerun()


    with st.expander("Backend Mode Rules"):
        st.markdown("""
        * **SQLite** remains the source of truth for metadata, ACL/RBAC, versioning, logs, and audit.
        * **Azure AI Search** stores searchable chunks only; governance remains controlled by the backend.
        * **Azure Blob Storage** stores uploaded source documents when configured.
        * **Azure OpenAI** can be used when endpoint, deployment, and key configuration are available.
        * **SharePoint** uses the governed source connector model; live enterprise sync depends on tenant configuration.
        """)