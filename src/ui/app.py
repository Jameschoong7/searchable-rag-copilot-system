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

from src.core.answer_status import classify_answer_status_detail
from src.core.user_repository import authenticate_user


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


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

UPLOAD_VALIDATE_URL = f"{API_BASE_URL}/admin/validate-upload"
REINDEX_URL = f"{API_BASE_URL}/admin/reindex"
API_URL = f"{API_BASE_URL}/query"
API_HEALTH_URL = f"{API_BASE_URL}/health"
CHAT_JOBS_URL = f"{API_BASE_URL}/chat/jobs"
CHAT_SESSIONS_URL = f"{API_BASE_URL}/chat/sessions"
BACKEND_JOBS_URL = f"{API_BASE_URL}/admin/jobs"
REINDEX_JOBS_URL = f"{API_BASE_URL}/admin/reindex-jobs"
INDEX_UPDATE_JOBS_URL = f"{API_BASE_URL}/admin/index-update-jobs"
INDEX_SNAPSHOT_JOBS_URL = f"{API_BASE_URL}/admin/index-snapshot-jobs"
METADATA_UPDATE_VALIDATE_URL = f"{API_BASE_URL}/admin/validate-metadata-update"
ARCHIVE_DOCUMENT_URL = f"{API_BASE_URL}/admin/archive-document"
UNARCHIVE_DOCUMENT_URL = f"{API_BASE_URL}/admin/unarchive-document"
ARCHIVE_DOCUMENT_JOB_URL = f"{API_BASE_URL}/admin/archive-document-jobs"
UNARCHIVE_DOCUMENT_JOB_URL = f"{API_BASE_URL}/admin/unarchive-document-jobs"
SETTINGS_URL = f"{API_BASE_URL}/admin/settings"
UPLOAD_DOCUMENT_URL = f"{API_BASE_URL}/admin/upload-document"
UPLOAD_DOCUMENT_VERSION_URL = f"{API_BASE_URL}/admin/upload-document-version"
UPLOAD_ZIP_STAGING_URL = f"{API_BASE_URL}/admin/upload-zip-staging"
APPROVE_DOCUMENT_URL = f"{API_BASE_URL}/admin/approve-document"
REJECT_STAGED_DOCUMENT_URL = f"{API_BASE_URL}/admin/reject-staged-document"
QUERY_LOG_DB_PATH = PROJECT_ROOT / "data/logs/query_logs.db"
ONEDRIVE_FILES_URL = f"{API_BASE_URL}/admin/graph/onedrive/files"
ONEDRIVE_STAGE_FILE_URL = f"{API_BASE_URL}/admin/graph/onedrive/stage-file"
ONEDRIVE_STAGE_FILES_JOB_URL = f"{API_BASE_URL}/admin/graph/onedrive/stage-files-job"
ONEDRIVE_REFRESH_FILE_URL = f"{API_BASE_URL}/admin/graph/onedrive/refresh-file"
ONEDRIVE_REFRESH_FILES_JOB_URL = f"{API_BASE_URL}/admin/graph/onedrive/refresh-files-job"
ONENOTE_PAGES_URL = f"{API_BASE_URL}/admin/graph/onenote/pages"
ONENOTE_STAGE_PAGES_JOB_URL = f"{API_BASE_URL}/admin/graph/onenote/stage-pages-job"
ONENOTE_REFRESH_PAGE_URL = f"{API_BASE_URL}/admin/graph/onenote/refresh-page"
ONENOTE_REFRESH_PAGES_JOB_URL = f"{API_BASE_URL}/admin/graph/onenote/refresh-pages-job"
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


def request_staged_document_rejection(document_id: str) -> dict:
    """Ask FastAPI to reject one pending-review connector document."""
    response = requests.post(
        REJECT_STAGED_DOCUMENT_URL,
        json={
            "role": st.session_state["role"],
            "user_department": st.session_state["department"],
            "document_id": document_id,
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


def submit_onedrive_stage_job(file_items: list[dict]) -> dict:
    """Submit selected OneDrive files as one durable backend staging job."""
    response = requests.post(
        ONEDRIVE_STAGE_FILES_JOB_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "files": [
                {
                    "item_id": file_item["id"],
                    "name": file_item["name"],
                    "connector_path": file_item["connector_path"],
                }
                for file_item in file_items
            ],
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def request_onedrive_file_refresh(file_item: dict) -> dict:
    """Ask FastAPI to refresh one already-ingested OneDrive source file."""
    response = requests.post(
        ONEDRIVE_REFRESH_FILE_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "user_department": st.session_state["department"],
            "item_id": file_item["id"],
            "name": file_item["name"],
            "connector_path": file_item["connector_path"],
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def submit_onedrive_refresh_job(file_items: list[dict]) -> dict:
    """Submit selected OneDrive files as one durable backend refresh job."""
    response = requests.post(
        ONEDRIVE_REFRESH_FILES_JOB_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "user_department": st.session_state["department"],
            "files": [
                {
                    "item_id": file_item["id"],
                    "name": file_item["name"],
                    "connector_path": file_item["connector_path"],
                }
                for file_item in file_items
            ],
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def request_onenote_page_scan() -> dict:
    """Ask FastAPI to list pages under the configured OneNote notebook scope."""
    response = requests.post(
        ONENOTE_PAGES_URL,
        json={
            "role": st.session_state["role"],
        },
        timeout=60,
    )

    response.raise_for_status()
    return response.json()


def submit_onenote_stage_job(page_items: list[dict]) -> dict:
    """Submit selected OneNote pages as one durable backend staging job."""
    response = requests.post(
        ONENOTE_STAGE_PAGES_JOB_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "pages": [
                {
                    "page_id": page_item["id"],
                    "title": page_item.get("title") or "Untitled Page",
                    "connector_path": page_item["connector_path"],
                }
                for page_item in page_items
            ],
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def request_onenote_page_refresh(page_item: dict) -> dict:
    """Ask FastAPI to refresh one already-ingested OneNote source page."""
    response = requests.post(
        ONENOTE_REFRESH_PAGE_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "user_department": st.session_state["department"],
            "page_id": page_item["id"],
            "title": page_item.get("title") or "Untitled Page",
            "connector_path": page_item["connector_path"],
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def submit_onenote_refresh_job(page_items: list[dict]) -> dict:
    """Submit selected OneNote pages as one durable backend refresh job."""
    response = requests.post(
        ONENOTE_REFRESH_PAGES_JOB_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "user_department": st.session_state["department"],
            "pages": [
                {
                    "page_id": page_item["id"],
                    "title": page_item.get("title") or "Untitled Page",
                    "connector_path": page_item["connector_path"],
                }
                for page_item in page_items
            ],
        },
        timeout=10,
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


def submit_index_snapshot_job() -> dict:
    """Submit a saved index snapshot refresh as a durable backend job."""
    response = requests.post(
        INDEX_SNAPSHOT_JOBS_URL,
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
    """Submit a durable archive job for one document."""
    response = requests.post(
        ARCHIVE_DOCUMENT_JOB_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "user_department": st.session_state["department"],
            "document_id": document_id,
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def request_document_unarchive(document_id: str) -> dict:
    """Submit a durable restore job for one manually archived document."""
    response = requests.post(
        UNARCHIVE_DOCUMENT_JOB_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "user_department": st.session_state["department"],
            "document_id": document_id,
        },
        timeout=10,
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


def request_backend_zip_staging(uploaded_zip_file) -> dict:
    """Ask FastAPI to extract a ZIP and stage supported files for review."""
    response = requests.post(
        UPLOAD_ZIP_STAGING_URL,
        data={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
            "user_department": st.session_state["department"],
        },
        files={
            "file": (
                uploaded_zip_file.name,
                uploaded_zip_file.getvalue(),
                uploaded_zip_file.type or "application/zip",
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
            "session_id": st.session_state.get("chat_session_id"),
            "use_memory": st.session_state.get("chat_memory_enabled", True),
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def get_backend_chat_sessions() -> list[dict]:
    """Load recent persisted chat sessions for the signed-in user."""
    response = requests.get(
        CHAT_SESSIONS_URL,
        params={"user": st.session_state["user"]},
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def get_backend_chat_session_messages(session_id: str) -> list[dict]:
    """Load persisted messages for one selected chat session."""
    response = requests.get(
        f"{CHAT_SESSIONS_URL}/{session_id}/messages",
        params={"user": st.session_state["user"]},
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def convert_persisted_message_to_chat_message(message: dict) -> dict:
    """Convert a backend chat-memory row into Streamlit chat display state."""
    if message["message_role"] == "user":
        return {
            "role": "user",
            "content": message["content"],
        }

    context_text = (
        "Loaded from persistent chat memory. "
        "Follow-up answers still use current ACL-filtered retrieval."
    )

    return {
        "role": "assistant",
        "content": message["content"],
        "sources": message.get("sources", []),
        "context": context_text,
        "status": message.get("status") or "success",
        "feedback": "none",
    }


def get_backend_job(job_id: str) -> dict:
    """Poll one backend job by ID."""
    response = requests.get(
        f"{BACKEND_JOBS_URL}/{job_id}",
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def get_latest_backend_job(job_type: str) -> dict | None:
    """Load the latest backend job of one type for dashboard evidence."""
    response = requests.get(
        f"{BACKEND_JOBS_URL}/latest",
        params={"job_type": job_type},
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
        if result.get("session_id"):
            st.session_state["chat_session_id"] = result["session_id"]

        answer_status_detail = classify_answer_status_detail(
            result["answer"],
            result["sources"],
        )
        answer_status = answer_status_detail["status"]

        query_log_id = write_query_log(
            question=result["question"],
            department_filter=result.get("department_filter"),
            file_type_filter=result.get("file_type_filter"),
            status=answer_status,
            status_reason=answer_status_detail["reason"],
            answer_text=result["answer"],
            sources=result["sources"],
            latency_seconds=result.get("latency_seconds", 0),
        )

        context_text = (
            f"Access context: {result['role']} / {result['department']} | "
            f"Search department: {result.get('department_filter') or 'ACL-permitted shared scope'} | "
            f"File type: {result.get('file_type_filter')} | "
            f"Memory: {'On' if result.get('use_memory', True) else 'Off'} | "
            f"LLM: {result.get('llm_backend', 'unknown')} / "
            f"{result.get('llm_deployment', 'unknown')}"
        )
        retrieval_question = result.get("retrieval_question")

        if retrieval_question and retrieval_question != result["question"]:
            context_text = (
                f"{context_text} | "
                f"Retrieval question: {retrieval_question}"
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
        if result.get("session_id"):
            st.session_state["chat_session_id"] = result["session_id"]

        query_log_id = write_query_log(
            question=result.get("question", "Unknown question"),
            department_filter=result.get("department_filter"),
            file_type_filter=result.get("file_type_filter"),
            status="api_error",
            status_reason="Backend chat job failed",
            answer_text=job["message"],
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


@st.fragment(run_every="2s")
def poll_active_index_snapshot_job() -> None:
    """Poll the active index snapshot refresh job."""
    active_index_snapshot_job_id = st.session_state.get("active_index_snapshot_job_id")

    if not active_index_snapshot_job_id:
        return

    try:
        job = get_backend_job(active_index_snapshot_job_id)
    except requests.exceptions.RequestException as error:
        st.warning(f"Index snapshot job status unavailable: {error}")
        return

    if job["status"] in ["queued", "running"]:
        st.info(job["message"])
        return

    if job["status"] == "succeeded":
        result = job["result"]
        st.session_state["index_snapshot_job_message"] = result.get(
            "message",
            job["message"],
        )
        st.session_state["index_snapshot_job_status"] = "success"

    elif job["status"] == "failed":
        st.session_state["index_snapshot_job_message"] = job["message"]
        st.session_state["index_snapshot_job_status"] = "error"

    st.session_state.pop("active_index_snapshot_job_id", None)
    st.rerun()


@st.fragment(run_every="2s")
def poll_active_onedrive_stage_job() -> None:
    """Poll the active OneDrive staging job without blocking Streamlit navigation."""
    active_job_id = st.session_state.get("active_onedrive_stage_job_id")

    if not active_job_id:
        return

    try:
        job = get_backend_job(active_job_id)
    except requests.exceptions.RequestException as error:
        st.warning(f"OneDrive staging job status unavailable: {error}")
        return

    if job["status"] in ["queued", "running"]:
        st.info(job["message"])
        return

    if job["status"] == "succeeded":
        result = job["result"]
        st.session_state["onedrive_stage_results"] = result.get("results", [])
        st.session_state["onedrive_stage_message"] = result.get(
            "message",
            job["message"],
        )
        st.session_state["onedrive_stage_status"] = "success"

    elif job["status"] == "failed":
        st.session_state["onedrive_stage_message"] = job["message"]
        st.session_state["onedrive_stage_status"] = "error"

    st.session_state.pop("active_onedrive_stage_job_id", None)
    st.rerun()


@st.fragment(run_every="2s")
def poll_active_onedrive_refresh_job() -> None:
    """Poll the active OneDrive refresh job without blocking Streamlit navigation."""
    active_job_id = st.session_state.get("active_onedrive_refresh_job_id")

    if not active_job_id:
        return

    try:
        job = get_backend_job(active_job_id)
    except requests.exceptions.RequestException as error:
        st.warning(f"OneDrive refresh job status unavailable: {error}")
        return

    if job["status"] in ["queued", "running"]:
        st.info(job["message"])
        return

    if job["status"] == "succeeded":
        result = job["result"]
        st.session_state["onedrive_refresh_results"] = result.get("results", [])
        st.session_state["onedrive_refresh_message"] = result.get(
            "message",
            job["message"],
        )
        st.session_state["onedrive_refresh_status"] = "success"

    elif job["status"] == "failed":
        st.session_state["onedrive_refresh_message"] = job["message"]
        st.session_state["onedrive_refresh_status"] = "error"

    st.session_state.pop("active_onedrive_refresh_job_id", None)
    st.rerun()


@st.fragment(run_every="2s")
def poll_active_onenote_stage_job() -> None:
    """Poll the active OneNote staging job without blocking Streamlit navigation."""
    active_job_id = st.session_state.get("active_onenote_stage_job_id")

    if not active_job_id:
        return

    try:
        job = get_backend_job(active_job_id)
    except requests.exceptions.RequestException as error:
        st.warning(f"OneNote staging job status unavailable: {error}")
        return

    if job["status"] in ["queued", "running"]:
        st.info(job["message"])
        return

    if job["status"] == "succeeded":
        result = job["result"]
        st.session_state["onenote_stage_results"] = result.get("results", [])
        st.session_state["onenote_stage_message"] = result.get(
            "message",
            job["message"],
        )
        st.session_state["onenote_stage_status"] = "success"

    elif job["status"] == "failed":
        st.session_state["onenote_stage_message"] = job["message"]
        st.session_state["onenote_stage_status"] = "error"

    st.session_state.pop("active_onenote_stage_job_id", None)
    st.rerun()


@st.fragment(run_every="2s")
def poll_active_onenote_refresh_job() -> None:
    """Poll the active OneNote refresh job without blocking Streamlit navigation."""
    active_job_id = st.session_state.get("active_onenote_refresh_job_id")

    if not active_job_id:
        return

    try:
        job = get_backend_job(active_job_id)
    except requests.exceptions.RequestException as error:
        st.warning(f"OneNote refresh job status unavailable: {error}")
        return

    if job["status"] in ["queued", "running"]:
        st.info(job["message"])
        return

    if job["status"] == "succeeded":
        result = job["result"]
        st.session_state["onenote_refresh_results"] = result.get("results", [])
        st.session_state["onenote_refresh_message"] = result.get(
            "message",
            job["message"],
        )
        st.session_state["onenote_refresh_status"] = "success"

    elif job["status"] == "failed":
        st.session_state["onenote_refresh_message"] = job["message"]
        st.session_state["onenote_refresh_status"] = "error"

    st.session_state.pop("active_onenote_refresh_job_id", None)
    st.rerun()


@st.fragment(run_every="2s")
def poll_active_document_lifecycle_job() -> None:
    """Poll active archive/restore jobs without blocking Streamlit navigation."""
    active_job_id = st.session_state.get("active_document_lifecycle_job_id")

    if not active_job_id:
        return

    try:
        job = get_backend_job(active_job_id)
    except requests.exceptions.RequestException as error:
        st.warning(f"Document lifecycle job status unavailable: {error}")
        return

    if job["status"] in ["queued", "running"]:
        st.info(job["message"])
        return

    if job["status"] == "succeeded":
        result = job["result"]
        lifecycle_message = result.get(
            "message",
            job["message"],
        )

        if job.get("job_type") == "document_unarchive":
            lifecycle_message = (
                f"{lifecycle_message} Run Update for Pending Documents to make it searchable."
            )

        st.session_state["document_lifecycle_message"] = lifecycle_message
        st.session_state["document_lifecycle_status"] = "success"

    elif job["status"] == "failed":
        st.session_state["document_lifecycle_message"] = job["message"]
        st.session_state["document_lifecycle_status"] = "error"

    st.session_state.pop("active_document_lifecycle_job_id", None)
    st.rerun()


def render_sidebar_notice_messages(pending_index_count: int = 0) -> None:
    """Show durable workflow results without stacking loud success banners."""
    st.markdown('<div class="sidebar-section-label">Notices</div>', unsafe_allow_html=True)

    notice_specs = [
        ("document_lifecycle_message", "document_lifecycle_status"),
        ("index_snapshot_job_message", "index_snapshot_job_status"),
        ("index_update_job_message", "index_update_job_status"),
        ("reindex_job_message", "reindex_job_status"),
    ]
    active_notice_specs = [
        (message_key, status_key)
        for message_key, status_key in notice_specs
        if st.session_state.get(message_key)
    ]

    if not active_notice_specs:
        st.caption("No active notices.")
        return

    action_notices = []
    in_progress_notices = []
    activity_notices = []

    for message_key, status_key in active_notice_specs:
        message = st.session_state[message_key]
        status = st.session_state.get(status_key, "info")

        needs_index_update = (
            "marked pending index" in message
            or "Run Update for Pending Documents" in message
        )

        if needs_index_update and pending_index_count == 0:
            st.session_state.pop(message_key, None)
            st.session_state.pop(status_key, None)
            continue

        if status == "error":
            action_notices.append(("error", message))
        elif needs_index_update:
            action_notices.append(("warning", message))
        elif status == "success":
            activity_notices.append(message)
        else:
            in_progress_notices.append(message)

    if not action_notices and not in_progress_notices and not activity_notices:
        st.caption("No active notices.")
        return

    for notice_type, message in action_notices:
        if notice_type == "error":
            st.error(message)
        else:
            st.warning(message)

    for message in in_progress_notices:
        st.info(message)

    if activity_notices:
        st.caption("Latest activity")
        st.markdown(
            f"""
            <div style="
                padding: 0.55rem 0.7rem;
                border: 1px solid #e2e8f0;
                border-radius: 0.5rem;
                background: #ffffff;
                color: #475569;
                font-size: 0.82rem;
                line-height: 1.45;
            ">{escape(activity_notices[0])}</div>
            """,
            unsafe_allow_html=True,
        )

        if len(activity_notices) > 1:
            with st.expander("Recent activity", expanded=False):
                for message in activity_notices[1:]:
                    st.caption(message)

    if st.button("Clear notices", use_container_width=True, key="clear_sidebar_notices"):
        for message_key, status_key in notice_specs:
            st.session_state.pop(message_key, None)
            st.session_state.pop(status_key, None)

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
    """Check whether the current Streamlit session has an authenticated user."""
    return "user" in st.session_state


def login_user(username: str, password: str) -> bool:
    """Validate a SQLite-backed user and store role/department in the session."""
    account = authenticate_user(username, password)

    if account is None:
        return False

    st.session_state["user"] = account["username"]
    st.session_state["role"] = account["role"]
    st.session_state["department"] = account["department"]
    return True


def logout_user() -> None:
    """Clear authentication state from the current Streamlit session."""
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


def summarize_connector_refresh_job(job: dict | None, connector_label: str) -> dict:
    """Return compact measured counts from the latest connector refresh job."""
    if not job:
        return {
            "Connector": connector_label,
            "Checked": 0,
            "Updated": 0,
            "Unchanged": 0,
            "Rejected / Error": 0,
            "Status": "No job",
            "Updated At": "-",
        }

    result = job.get("result") or {}
    result_rows = result.get("results", [])
    rejected_or_error_count = sum(
        1
        for row in result_rows
        if row.get("Status") in ["Rejected", "Error"]
    )

    return {
        "Connector": connector_label,
        "Checked": result.get("total_count", len(result_rows)),
        "Updated": result.get("updated_count", 0),
        "Unchanged": result.get("unchanged_count", 0),
        "Rejected / Error": rejected_or_error_count,
        "Status": job.get("status", "unknown").title(),
        "Updated At": job.get("updated_at", "-"),
    }


def is_connector_document(document: dict) -> bool:
    """Check whether a metadata row came from a live Graph connector."""
    return document.get("source") in ["onedrive", "onenote"]


def build_connector_version_chain_rows(documents: list[dict]) -> list[dict]:
    """Build a compact previous-version to new-version table for Graph connectors."""
    document_by_id = {
        document["document_id"]: document
        for document in documents
    }
    chain_rows = []

    for previous_document in documents:
        replacement_id = previous_document.get("replaced_by_document_id")

        if (
            not is_connector_document(previous_document)
            or previous_document.get("chunk_id") != "archived"
            or not replacement_id
        ):
            continue

        new_document = document_by_id.get(replacement_id, {})
        source_label = (
            new_document.get("source")
            or previous_document.get("source")
            or "connector"
        ).title()

        chain_rows.append(
            {
                "Connector": source_label,
                "Changed Source": new_document.get("title") or previous_document.get("title"),
                "Previous Document ID": previous_document["document_id"],
                "New Document ID": replacement_id,
                "Version": get_version_label(new_document) if new_document else "-",
                "Index Status": get_index_status_label(new_document) if new_document else "Missing",
                "Archived At": previous_document.get("archived_at") or "-",
            }
        )

    return sorted(
        chain_rows,
        key=lambda row: row["Archived At"],
        reverse=True,
    )


def get_pending_connector_versions(documents: list[dict]) -> list[dict]:
    """Return connector-backed active versions waiting for index update."""
    return [
        document
        for document in documents
        if (
            is_connector_document(document)
            and document.get("is_active") == 1
            and document.get("chunk_id") in ["pending", "pending_index"]
        )
    ]


def get_pending_index_documents(documents: list[dict]) -> list[dict]:
    """Return active visible documents waiting for index update."""
    return [
        document
        for document in documents
        if (
            document.get("is_active") == 1
            and document.get("chunk_id") in ["pending", "pending_index"]
        )
    ]


def render_pending_index_notice(
    pending_documents: list[dict],
    label: str = "document",
    expander_label: str = "Pending Documents",
    show_index_sync_hint: bool = True,
    use_expander: bool = True,
) -> None:
    """Show pending-index follow-up guidance without changing backend state."""
    if not pending_documents:
        return

    st.warning(
        f"{len(pending_documents)} {label}(s) are waiting for index update."
    )

    if show_index_sync_hint:
        st.caption(
            "Open Index Sync and run Update for Pending Documents so chat uses the refreshed content."
        )

    table_rows = [
        {
            "Title": document.get("title"),
            "Source": document.get("source", ""),
            "Department": document.get("department"),
            "Document ID": document.get("document_id"),
            "Version": get_version_label(document),
            "Index Status": get_index_status_label(document),
            "Updated At": document.get("uploaded_at"),
        }
        for document in pending_documents
    ]

    if use_expander:
        with st.expander(expander_label, expanded=False):
            st.dataframe(
                table_rows,
                use_container_width=True,
                hide_index=True,
                height=min(260, 38 * (len(pending_documents) + 1)),
            )
    else:
        st.dataframe(
            table_rows,
            use_container_width=True,
            hide_index=True,
            height=min(260, 38 * (len(pending_documents) + 1)),
        )


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
            "answer_text": "TEXT",
            "status_reason": "TEXT",
        }

        for column_name, column_type in feedback_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE query_logs ADD COLUMN {column_name} {column_type}"
                )


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


def format_logged_sources(status: str, sources_json: str) -> str:
    """Format logged source paths according to whether they supported the answer."""
    try:
        sources = json.loads(sources_json)
    except json.JSONDecodeError:
        sources = []

    if status == "permission_block":
        return "No sources used"

    if not sources:
        return "None"

    source_text = ", ".join(sources)

    if status == "not_found":
        return f"Checked: {source_text}"

    return f"Sources: {source_text}"


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
        status_reason: str,
        answer_text: str,
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
                status_reason,
                answer_text,
                sources_json,
                latency_seconds
            )
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                status_reason,
                answer_text,
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
            (QUERY_HISTORY_LIMIT,),
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
            "query_history_limit": QUERY_HISTORY_LIMIT,
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

    if document.get("chunk_id") == "rejected":
        return "Rejected"

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


def render_status_card(label: str, value: int | str, tone: str = "neutral") -> None:
    """Render a compact status card with stronger attention colors than st.metric."""
    tones = {
        "neutral": {
            "background": "#ffffff",
            "border": "#d0d5dd",
            "label": "#667085",
            "value": "#101828",
            "bar": "#98a2b3",
        },
        "attention": {
            "background": "#fff7ed",
            "border": "#fdba74",
            "label": "#9a3412",
            "value": "#7c2d12",
            "bar": "#f97316",
        },
        "danger": {
            "background": "#fff1f0",
            "border": "#f6b3ae",
            "label": "#b42318",
            "value": "#7a271a",
            "bar": "#e63329",
        },
        "success": {
            "background": "#f0fdf4",
            "border": "#86efac",
            "label": "#166534",
            "value": "#14532d",
            "bar": "#22c55e",
        },
        "info": {
            "background": "#eff6ff",
            "border": "#93c5fd",
            "label": "#1d4ed8",
            "value": "#1e3a8a",
            "bar": "#3b82f6",
        },
    }
    selected_tone = tones.get(tone, tones["neutral"])

    st.markdown(
        f"""
        <div style="
            background:{selected_tone['background']};
            border:1px solid {selected_tone['border']};
            border-left:4px solid {selected_tone['bar']};
            border-radius:8px;
            padding:0.75rem 0.9rem;
            box-shadow:0 1px 2px rgba(16,24,40,0.05);
        ">
            <div style="
                color:{selected_tone['label']};
                font-size:0.75rem;
                font-weight:750;
                text-transform:uppercase;
                letter-spacing:0.02em;
            ">{escape(str(label))}</div>
            <div style="
                color:{selected_tone['value']};
                font-size:1.55rem;
                font-weight:800;
                line-height:1.2;
                margin-top:0.15rem;
            ">{escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_header(title: str, subtitle: str) -> None:
    """Render the standard header used by KB Management workflow tabs."""
    st.markdown(
        f"""
        <div class="workflow-header">
            <div class="workflow-title">{escape(title)}</div>
            <div class="workflow-subtitle">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    st.caption("Use an assigned portal account. Registration is disabled.")

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
brand_red = "#e63329"
brand_red_hover = "#c42820"
brand_red_soft = "#fff1f0"
brand_red_border = "#f6b3ae"

# Global CSS updates
st.markdown(
    f"""
    <style>
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stActionButton"],
    .stAppDeployButton {{
        display: none !important;
    }}

    .stApp {{
        background-color: #f6f7f9;
        color: #101828;
    }}

    .block-container {{
        padding-top: 0.8rem;
        max-width: 1480px;
    }}

    [data-testid="stSidebar"] {{
        background-color: #ffffff;
        border-right: 1px solid #d0d5dd;
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
        background-color: {brand_red_soft};
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
        background-color: {brand_red};
        color: #ffffff !important;
        font-weight: 600;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {{
        color: #ffffff !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked):hover {{
        background-color: {brand_red_hover};
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

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: #cbd5e1 !important;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
        margin-bottom: 1rem;
    }}

    .element-container:has(> div[data-testid="stVerticalBlockBorderWrapper"]) {{
        margin-bottom: 1rem !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] + div[data-testid="stVerticalBlockBorderWrapper"] {{
        margin-top: 0.35rem;
    }}

    div[data-testid="stTabs"] [role="tablist"] {{
        gap: 0.35rem;
        border-bottom: 1px solid #d0d5dd;
        margin-bottom: 0.85rem;
    }}

    div[data-testid="stTabs"] button[role="tab"] {{
        min-height: 2.5rem;
        padding: 0.55rem 0.9rem;
        border: 1px solid #d0d5dd;
        border-bottom-color: transparent;
        border-radius: 8px 8px 0 0;
        background: #f9fafb;
        color: #475467 !important;
        font-size: 0.9rem;
        font-weight: 750;
        letter-spacing: 0;
    }}

    div[data-testid="stTabs"] button[role="tab"] p {{
        color: inherit !important;
        font-size: inherit;
        font-weight: inherit;
    }}

    div[data-testid="stTabs"] button[role="tab"]:hover {{
        background: {brand_red_soft};
        color: {brand_red_hover} !important;
    }}

    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        background: #ffffff;
        color: {brand_red} !important;
        border-color: {brand_red_border};
        border-bottom-color: #ffffff;
        box-shadow: inset 0 3px 0 {brand_red};
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid #d0d5dd;
        border-radius: 8px;
        overflow: hidden;
    }}

    div[data-testid="stExpander"] {{
        margin: 0.85rem 0 1rem 0;
    }}

    .stButton > button {{
        border-radius: 7px;
        border-color: #98a2b3;
        font-weight: 650;
    }}

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {{
        background-color: {brand_red};
        border-color: {brand_red};
        color: #ffffff;
    }}

    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primary"]:hover {{
        background-color: {brand_red_hover};
        border-color: {brand_red_hover};
        color: #ffffff;
    }}

    .stButton > button:focus,
    .stFormSubmitButton > button:focus,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within {{
        border-color: {brand_red} !important;
        box-shadow: 0 0 0 1px {brand_red_border} !important;
    }}

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {{
        background-color: #ffffff !important;
        border-color: #98a2b3 !important;
    }}

    input,
    textarea {{
        color: #101828 !important;
    }}

    input::placeholder,
    textarea::placeholder {{
        color: #667085 !important;
        opacity: 1 !important;
    }}

    div[data-baseweb="input"][aria-disabled="true"] > div,
    input:disabled,
    textarea:disabled {{
        background-color: #f2f4f7 !important;
        color: #475467 !important;
        -webkit-text-fill-color: #475467 !important;
    }}

    div[data-testid="stMetric"] {{
        background-color: #ffffff;
        border: 1px solid #d0d5dd;
        padding: 14px;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
    }}

    .element-container:has(.chat-history-panel-marker) + .element-container div[data-testid="stVerticalBlockBorderWrapper"],
    .element-container:has(.chat-main-panel-marker) + .element-container div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: #ffffff !important;
        border-color: #b8c0cc !important;
        box-shadow: 0 10px 26px rgba(16, 24, 40, 0.08) !important;
    }}

    .element-container:has(.chat-main-panel-marker) + .element-container div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-top: 3px solid {brand_red} !important;
    }}

    .element-container:has(.chat-message-panel-marker) + .element-container div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: #eef2f7 !important;
        border-color: #c7d0dd !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75) !important;
    }}

    .compact-section-title {{
        margin: 0.25rem 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 750;
        color: #101828;
    }}

    .workflow-header {{
        border-bottom: 1px solid #e4e7ec;
        margin: -0.1rem 0 1rem 0;
        padding: 0 0 0.75rem 0;
    }}

    .workflow-title {{
        color: #101828;
        font-size: 1.02rem;
        font-weight: 800;
        line-height: 1.3;
        margin: 0;
    }}

    .workflow-subtitle {{
        color: #667085;
        font-size: 0.84rem;
        line-height: 1.45;
        margin-top: 0.2rem;
        max-width: 72ch;
    }}

    .muted-note {{
        color: #667085;
        font-size: 0.84rem;
    }}

    .status-pill {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        font-size: 0.74rem;
        font-weight: 750;
        border: 1px solid transparent;
    }}

    .sidebar-section-label {{
        color: #667085;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 0.25rem 0 0.45rem 0;
    }}

    .account-card {{
        padding: 0.8rem;
        border: 1px solid #d0d5dd;
        border-left: 4px solid {brand_red};
        border-radius: 0.5rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(16,24,40,0.05);
        margin-bottom: 1rem;
    }}

    .account-name {{
        color: #101828;
        font-size: 0.95rem;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 0.55rem;
        overflow-wrap: anywhere;
    }}

    .account-meta-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }}

    .account-chip {{
        display: inline-flex;
        align-items: center;
        border: 1px solid #d0d5dd;
        border-radius: 999px;
        padding: 0.18rem 0.5rem;
        color: #344054;
        background: #f9fafb;
        font-size: 0.72rem;
        font-weight: 750;
        line-height: 1.25;
    }}

    .typing-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        color: #475467;
        font-size: 0.92rem;
        line-height: 1.5;
    }}

    .typing-dot {{
        width: 0.38rem;
        height: 0.38rem;
        border-radius: 999px;
        background: {brand_red};
        animation: typingPulse 1.1s infinite ease-in-out;
    }}

    .typing-dot:nth-child(2) {{
        animation-delay: 0.16s;
    }}

    .typing-dot:nth-child(3) {{
        animation-delay: 0.32s;
    }}

    @keyframes typingPulse {{
        0%, 80%, 100% {{
            opacity: 0.28;
            transform: translateY(0);
        }}
        40% {{
            opacity: 1;
            transform: translateY(-3px);
        }}
    }}

    * {{
        box-sizing: border-box;
    }}

    div[data-testid="column"] {{
        min-width: 0;
    }}

    div[data-testid="stDataFrame"] {{
        max-width: 100%;
        overflow-x: auto;
    }}

    @media (max-width: 980px) {{
        .block-container {{
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            max-width: 100%;
        }}

        div[data-testid="column"] {{
            flex: 1 1 100% !important;
            width: 100% !important;
        }}

        .status-pill {{
            width: 100%;
            justify-content: center;
            margin-top: 0.35rem;
        }}

        .stButton > button,
        .stFormSubmitButton > button {{
            min-height: 2.45rem;
            white-space: normal;
        }}
    }}

    @media (max-width: 640px) {{
        .block-container {{
            padding-top: 0.45rem;
        }}

        h1 {{
            font-size: 1.55rem !important;
        }}

        h2, h3 {{
            font-size: 1.12rem !important;
        }}

        div[data-testid="stMetric"] {{
            padding: 10px;
        }}

        div[data-testid="stDataFrame"] {{
            font-size: 0.82rem;
        }}
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

st.sidebar.markdown(
    f"""
    <div class="sidebar-section-label">Account</div>
    <div class="account-card">
        <div class="account-name">{escape(st.session_state["user"])}</div>
        <div class="account-meta-row">
            <span class="account-chip">{escape(st.session_state["role"])}</span>
            <span class="account-chip">Dept: {escape(st.session_state["department"])}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

kb_page_label = get_kb_page_label()

page_options = [
    kb_page_label,
    "Chat",
]

if st.session_state["role"] in [SYSTEM_ADMIN_ROLE, PROJECT_MANAGER_ROLE]:
    page_options.insert(0, "Performance")

if can_access_settings():
    page_options.append("Settings")

if "selected_navigation_page" not in st.session_state:
    st.session_state["selected_navigation_page"] = kb_page_label

if st.session_state["selected_navigation_page"] not in page_options:
    st.session_state["selected_navigation_page"] = kb_page_label

selected_page = st.sidebar.radio(
    "Navigation",
    page_options,
    key="selected_navigation_page",
)

previous_selected_page = st.session_state.get("previous_selected_navigation_page")

if previous_selected_page != selected_page:
    st.session_state["previous_selected_navigation_page"] = selected_page

    for transient_key in [
        "selected_onedrive_files_to_stage",
        "selected_onedrive_files_to_refresh",
        "selected_onedrive_file_to_refresh",
        "selected_onenote_pages_to_stage",
        "selected_onenote_pages_to_refresh",
        "selected_onenote_page_to_refresh",
        "selected_connector_review_document",
    ]:
        st.session_state.pop(transient_key, None)

sidebar_pending_index_documents = []

if st.session_state["role"] in [SYSTEM_ADMIN_ROLE, PROJECT_MANAGER_ROLE]:
    sidebar_pending_index_documents = [
        document
        for document in load_document_metadata()
        if can_view_document(document) and get_index_status_label(document) == "Pending Index"
    ]

st.sidebar.divider()

with st.sidebar:
    poll_active_reindex_job()
    poll_active_index_update_job()
    poll_active_onedrive_stage_job()
    poll_active_onedrive_refresh_job()
    poll_active_onenote_stage_job()
    poll_active_onenote_refresh_job()
    poll_active_index_snapshot_job()
    poll_active_document_lifecycle_job()

    if (
        st.session_state["role"] == SYSTEM_ADMIN_ROLE
        and selected_page != kb_page_label
        and sidebar_pending_index_documents
    ):
        st.warning(
            f"{len(sidebar_pending_index_documents)} document(s) waiting for index update."
        )

    render_sidebar_notice_messages(
        pending_index_count=len(sidebar_pending_index_documents)
    )

st.sidebar.divider()

if st.sidebar.button("Logout", use_container_width=True):
    logout_user()
    st.rerun()


if selected_page == "Performance":
    st.header("Dashboard")

    documents = load_document_metadata()
    all_documents = load_document_metadata(include_inactive=True)
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
    connector_evidence_error = None
    latest_onedrive_refresh_job = None
    latest_onenote_refresh_job = None

    try:
        latest_onedrive_refresh_job = get_latest_backend_job("onedrive_refresh")
        latest_onenote_refresh_job = get_latest_backend_job("onenote_refresh")
    except requests.exceptions.RequestException as error:
        connector_evidence_error = str(error)

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
        st.subheader("System Overview")

        metric_columns = st.columns(4)

        with metric_columns[0]:
            render_status_card(
                "Time-To-First-Answer",
                f"{query_log_summary['average_latency']:.2f}s",
                "info",
            )

        with metric_columns[1]:
            render_status_card(
                "Eval Retrieval Accuracy",
                top_k_accuracy,
                "success" if evaluation_results else "neutral",
            )

        with metric_columns[2]:
            render_status_card(
                "Miss Rate",
                miss_rate,
                "attention" if evaluation_results and evaluation_summary["miss_count"] else "neutral",
            )

        with metric_columns[3]:
            render_status_card(
                "Indexed Documents",
                f"{indexed_document_count}",
                "neutral",
            )

    with st.container(border=True):
        st.subheader("Query Outcomes")

        live_metric_columns = st.columns(4)

        with live_metric_columns[0]:
            render_status_card(
                "Grounded Answers",
                query_log_summary["grounded_answers"],
                "success",
            )

        with live_metric_columns[1]:
            render_status_card(
                "Not Found",
                query_log_summary["not_found_queries"],
                "attention" if query_log_summary["not_found_queries"] else "neutral",
            )

        with live_metric_columns[2]:
            render_status_card(
                "Permission Blocks",
                query_log_summary["permission_blocks"],
                "info" if query_log_summary["permission_blocks"] else "neutral",
            )

        with live_metric_columns[3]:
            render_status_card(
                "API / Connection Errors",
                query_log_summary["error_queries"],
                "danger" if query_log_summary["error_queries"] else "neutral",
            )

        recent_outcome_rows = [
            {
                "Time": row[0],
                "User": row[1],
                "Role": row[2],
                "Department": row[3],
                "Question": row[4],
                "Status": get_status_label(row[7]),
                "Reason": row[8] or "",
                "Answer Preview": (row[9] or "")[:180],
                "Sources / Checked": format_logged_sources(row[7], row[10]),
                "Latency (s)": row[11],
                "Feedback": row[12] or "none",
            }
            for row in query_log_summary["recent_outcome_rows"]
        ]

        with st.expander("Recent Outcomes", expanded=False):
            if recent_outcome_rows:
                st.dataframe(
                    recent_outcome_rows,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No logged chat outcomes yet. Submit a Chat query to populate this table.")

    with st.container(border=True):
        st.subheader("Source Refresh")

        connector_active_documents = [
            document
            for document in documents
            if is_connector_document(document)
        ]
        connector_pending_index_documents = [
            document
            for document in connector_active_documents
            if document.get("chunk_id") in ["pending", "pending_index"]
        ]
        connector_version_chain_rows = build_connector_version_chain_rows(all_documents)
        connector_refresh_rows = [
            summarize_connector_refresh_job(
                latest_onedrive_refresh_job,
                "OneDrive",
            ),
            summarize_connector_refresh_job(
                latest_onenote_refresh_job,
                "OneNote",
            ),
        ]
        total_checked = sum(row["Checked"] for row in connector_refresh_rows)
        total_updated = sum(row["Updated"] for row in connector_refresh_rows)
        total_unchanged = sum(row["Unchanged"] for row in connector_refresh_rows)
        total_rejected_or_error = sum(row["Rejected / Error"] for row in connector_refresh_rows)

        update_metric_columns = st.columns(4)

        with update_metric_columns[0]:
            render_status_card("Sources Checked", total_checked)

        with update_metric_columns[1]:
            render_status_card(
                "Changed",
                total_updated,
                "attention" if total_updated else "neutral",
            )

        with update_metric_columns[2]:
            render_status_card(
                "Unchanged",
                total_unchanged,
                "neutral",
            )

        with update_metric_columns[3]:
            render_status_card(
                "Pending Index",
                len(connector_pending_index_documents),
                "attention" if connector_pending_index_documents else "success",
            )

        if connector_evidence_error:
            st.warning(
                f"Could not load latest connector job evidence from backend: {connector_evidence_error}"
            )

        with st.expander("Refresh Jobs", expanded=False):
            st.dataframe(
                connector_refresh_rows,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Version Changes", expanded=False):
            if connector_version_chain_rows:
                st.dataframe(
                    connector_version_chain_rows,
                    use_container_width=True,
                    hide_index=True,
                    height=min(260, 38 * (len(connector_version_chain_rows) + 1)),
                )
            else:
                st.caption("No source replacements recorded.")

        if total_rejected_or_error:
            st.warning(
                f"{total_rejected_or_error} connector refresh item(s) were rejected or errored in the latest jobs."
            )
        elif total_checked:
            st.caption(
                "Changed sources create pending versions. Unchanged sources are skipped."
            )
        else:
            st.caption(
                "Run a OneDrive or OneNote refresh to populate this section."
            )

    with st.container(border=True):
        index_status_columns = st.columns([3, 1])

        with index_status_columns[0]:
            st.subheader("Index Status")

        with index_status_columns[1]:
            if st.button(
                "Refresh",
                use_container_width=True,
                disabled=bool(st.session_state.get("active_index_snapshot_job_id")),
                key="refresh_index_snapshot_button",
                help="Refresh the saved index snapshot shown on this dashboard.",
            ):
                try:
                    job = submit_index_snapshot_job()
                except requests.exceptions.HTTPError as error:
                    st.session_state["index_snapshot_job_message"] = (
                        f"Snapshot refresh rejected by backend: {error.response.text}"
                    )
                    st.session_state["index_snapshot_job_status"] = "error"
                except requests.exceptions.RequestException as error:
                    st.session_state["index_snapshot_job_message"] = (
                        f"Could not submit snapshot refresh job: {error}"
                    )
                    st.session_state["index_snapshot_job_status"] = "error"
                else:
                    st.session_state["active_index_snapshot_job_id"] = job["job_id"]
                    st.session_state["index_snapshot_job_message"] = job["message"]
                    st.session_state["index_snapshot_job_status"] = "info"

                st.rerun()

        if index_benchmark_results:
            benchmark_snapshot = index_benchmark_results
            after_snapshot = benchmark_snapshot.get("after", benchmark_snapshot)
            benchmark_type = benchmark_snapshot.get("benchmark_type", "snapshot")

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
                    f"Indexed {changed_document_count} changed {document_label} in "
                    f"{elapsed_seconds}s. Updated {chunks_refreshed} chunks; skipped "
                    f"{avoided_chunks} unchanged chunks."
                )

                metric_columns = st.columns(4)

                with metric_columns[0]:
                    render_status_card(
                        "Changed Documents",
                        changed_document_count,
                        "attention" if changed_document_count else "neutral",
                    )

                with metric_columns[1]:
                    render_status_card(
                        "Chunks Updated",
                        chunks_refreshed,
                        "info" if chunks_refreshed else "neutral",
                    )

                with metric_columns[2]:
                    render_status_card(
                        "Vectors Removed",
                        deleted_vectors,
                        "attention" if deleted_vectors else "neutral",
                    )

                with metric_columns[3]:
                    active_index_delta = (
                        "Portal only"
                        if db_size_mb is None
                        else f"{db_size_mb} MB"
                    )

                    render_status_card(
                        "Active Index",
                        f"{active_vectors} vectors",
                        "neutral",
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

                with st.expander("Update Details", expanded=False):
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
                    render_status_card("Rebuild Time", f"{elapsed_seconds}s", "info")

                with metric_columns[1]:
                    render_status_card("Chunks Rebuilt", chunks_indexed, "attention")

                with metric_columns[2]:
                    render_status_card("Active Records", active_records)

                with metric_columns[3]:
                    active_index_delta = (
                        "Portal only"
                        if db_size_mb is None
                        else f"{db_size_mb} MB"
                    )

                    render_status_card(
                        "Active Index",
                        f"{active_vectors} vectors",
                        "neutral",
                    )

            else:
                metric_columns = st.columns(3)

                with metric_columns[0]:
                    render_status_card("Active Records", active_records)

                with metric_columns[1]:
                    render_status_card("Active Index", f"{active_vectors} vectors")

                with metric_columns[2]:
                    if db_size_mb is None:
                        render_status_card("Index Size", "Portal only")
                    else:
                        render_status_card("Index Size", f"{db_size_mb} MB")

            if archived_file_count:
                st.warning(
                    f"{archived_file_count} archived source file(s) remain on disk for audit, "
                    "but active-aware indexing excludes archived versions from the configured search index."
                )
            else:
                st.caption("Active records and source files are aligned.")

            with st.expander("Index Details", expanded=False):
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
        st.subheader("Query Latency")

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

    with st.expander("Evaluation Method"):
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

            st.markdown("**Threshold Comparison**")

            st.dataframe(
                threshold_interpretation["comparison_rows"],
                use_container_width=True,
                hide_index=True,
            )

            if threshold_interpretation["has_clear_winner"]:
                st.success(threshold_interpretation["recommendation"])
            else:
                st.info(threshold_interpretation["recommendation"])

    if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
        with st.expander("Benchmark Misses", expanded=False):
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
        f"Query Log - Latest {query_log_summary['query_history_limit']}",
        expanded=False,
    ):
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


    st.stop()


if selected_page in ["KB Management", "KB Status"]:
    title_columns = st.columns([2.6, 1])

    with title_columns[0]:
        st.header(selected_page)

    with title_columns[1]:
        st.markdown(
            f"""
            <div style="text-align:right; padding-top:0.35rem;">
                <span class="status-pill" style="
                    color:#344054;
                    background:#ffffff;
                    border-color:#d0d5dd;
                ">{escape(st.session_state["role"])} / {escape(st.session_state["department"])}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    active_documents = load_document_metadata()
    all_documents = load_document_metadata(include_inactive=True)

    visible_documents = [
        document for document in active_documents
        if can_view_document(document)
    ]


    restorable_archived_documents = [
        document
        for document in all_documents
        if (
            document.get("is_active") == 0
            and document.get("chunk_id") == "archived"
            and not document.get("replaced_by_document_id")
            and can_view_document(document)
        )
    ]

    pending_review_documents = [
        document
        for document in load_document_metadata(include_inactive=True)
        if document.get("chunk_id") == "pending_review"
    ]

    visible_pending_index_count = sum(
        1
        for document in visible_documents
        if get_index_status_label(document) == "Pending Index"
    )
    pending_index_documents = get_pending_index_documents(visible_documents)
    pending_connector_versions = get_pending_connector_versions(visible_documents)
    visible_indexed_count = sum(
        1
        for document in visible_documents
        if get_index_status_label(document) == "Indexed"
    )

    ocr_needed_count = sum(
        1
        for document in visible_documents
        if "OCR needed" in document.get("visual_extraction_status", "")
    )

    ocr_extracted_count = sum(
        1
        for document in visible_documents
        if "OCR text extracted" in document.get("visual_extraction_status", "")
    )

    if (
        visible_pending_index_count > 0
        and st.session_state.get("index_update_job_status") == "success"
        and not st.session_state.get("active_index_update_job_id")
    ):
        st.session_state.pop("index_update_job_message", None)
        st.session_state.pop("index_update_job_status", None)

    reviewable_count = sum(
        1
        for document in pending_review_documents
        if (
            st.session_state["role"] == SYSTEM_ADMIN_ROLE
            or document["department"] == st.session_state["department"]
        )
    )

    summary_columns = st.columns(5)

    with summary_columns[0]:
        render_status_card("Visible Docs", len(visible_documents))

    with summary_columns[1]:
        render_status_card(
            "Pending Review",
            reviewable_count,
            "danger" if reviewable_count else "neutral",
        )

    with summary_columns[2]:
        render_status_card(
            "Pending Index",
            visible_pending_index_count,
            "attention" if visible_pending_index_count else "neutral",
        )

    with summary_columns[3]:
        render_status_card("Indexed", visible_indexed_count, "success")

    with summary_columns[4]:
        render_status_card(
            "OCR Review",
            ocr_needed_count,
            "attention" if ocr_needed_count else "neutral",
        )

    if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
        if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
            upload_tab, onedrive_tab, onenote_tab, review_tab, index_tab = st.tabs(
                ["Upload", "OneDrive", "OneNote", "Review Queue", "Index Sync"]
            )
        else:
            upload_tab, review_tab = st.tabs(["Upload", "Review Queue"])
            onedrive_tab = None
            onenote_tab = None
            index_tab = None

        with upload_tab:
            if "upload_message" not in st.session_state:
                st.session_state["upload_message"] = ""

            if st.session_state["role"] in [SYSTEM_ADMIN_ROLE, PROJECT_MANAGER_ROLE]:
                with st.container(border=True):
                    render_workflow_header(
                        "Upload",
                        "Add new documents or replace an existing document version through the backend-controlled upload flow.",
                    )
                    new_document_tab, new_version_tab, batch_zip_tab = st.tabs(
                        ["Upload New Document", "Upload New Version", "Batch ZIP Upload"]
                    )
                    with new_document_tab:
                        st.markdown("**Upload & Categorize**")

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
                                        detail = (
                                            error.response.text
                                            if error.response is not None
                                            else str(error)
                                        )

                                        if error.response is not None and error.response.status_code == 409:
                                            st.warning(
                                                "A metadata record already exists for this filename. "
                                                "Use Upload New Version if this is an updated copy of an existing document, "
                                                "or rename the file before uploading it as a new document."
                                            )
                                        else:
                                            st.error(f"Upload rejected by backend: {detail}")
                                    except requests.exceptions.RequestException as error:
                                        st.error(f"Could not upload document through backend: {error}")
                                    else:
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
                        st.markdown("**Upload New Version**")

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

                    with batch_zip_tab:
                        st.markdown("**Batch ZIP Upload**")
                        uploaded_zip_file = st.file_uploader(
                            "Upload ZIP",
                            type=["zip"],
                            key=f"zip_upload_file_{upload_form_version}",
                        )

                        submitted_zip_upload = st.button(
                            "Stage ZIP for Review",
                            type="primary",
                            use_container_width=True,
                            disabled=uploaded_zip_file is None,
                            key=f"submit_zip_stage_{upload_form_version}",
                        )

                        if submitted_zip_upload:
                            try:
                                zip_result = request_backend_zip_staging(uploaded_zip_file)
                            except requests.exceptions.HTTPError as error:
                                st.error(f"ZIP staging rejected by backend: {error.response.text}")
                            except requests.exceptions.RequestException as error:
                                st.error(f"Could not stage ZIP upload: {error}")
                            else:
                                st.success(zip_result["message"])
                                if zip_result.get("results"):
                                    st.dataframe(
                                        zip_result["results"],
                                        use_container_width=True,
                                        hide_index=True,
                                        height=min(320, 38 * (len(zip_result["results"]) + 1)),
                                    )
                                if zip_result.get("staged_count", 0):
                                    st.info("Review Queue now contains the staged ZIP document(s).")
        if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
            with onedrive_tab:
                with st.container(border=True):
                    render_workflow_header(
                        "OneDrive",
                        "Scan the configured OneDrive knowledge folder, stage files for review, and refresh indexed source versions.",
                    )
                    connector_action_columns = st.columns([1, 2])

                    with connector_action_columns[0]:
                        if st.button(
                            "Scan OneDrive",
                            use_container_width=True,
                            key="scan_onedrive_root_button",
                        ):
                            try:
                                scan_result = request_onedrive_file_scan()
                            except requests.exceptions.HTTPError as error:
                                st.error(f"OneDrive scan rejected by backend: {error.response.text}")
                            except requests.exceptions.RequestException as error:
                                st.error(f"Could not scan OneDrive connector: {error}")
                            else:
                                st.session_state["onedrive_files"] = scan_result["files"]
                                st.success(f"Found {len(scan_result['files'])} file(s).")

                    with connector_action_columns[1]:
                        if st.session_state.get("onedrive_stage_message"):
                            stage_status = st.session_state.get("onedrive_stage_status", "info")

                            if stage_status == "success":
                                st.success(st.session_state["onedrive_stage_message"])
                            elif stage_status == "error":
                                st.error(st.session_state["onedrive_stage_message"])
                            else:
                                st.info(st.session_state["onedrive_stage_message"])

                    onedrive_files = st.session_state.get("onedrive_files", [])
                    file_options = {
                        f"{file_item['name']} - {file_item['connector_path']}": file_item
                        for file_item in onedrive_files
                    }

                    if onedrive_files:
                        stageable_count = sum(
                            1
                            for file_item in onedrive_files
                            if file_item.get("connector_state", "New") in ["New", "Rejected"]
                        )
                        refreshable_count = sum(
                            1
                            for file_item in onedrive_files
                            if file_item.get("connector_state") in ["Pending Index", "Indexed"]
                        )
                        connector_metric_columns = st.columns(3)

                        with connector_metric_columns[0]:
                            render_status_card("Discovered", len(onedrive_files))

                        with connector_metric_columns[1]:
                            render_status_card(
                                "Stageable",
                                stageable_count,
                                "attention" if stageable_count else "neutral",
                            )

                        with connector_metric_columns[2]:
                            render_status_card("Refreshable", refreshable_count)

                        with st.expander("Discovered Files", expanded=False):
                            st.dataframe(
                                [
                                    {
                                        "Name": file_item["name"],
                                        "State": file_item.get("connector_state", "New"),
                                        "Path": file_item["connector_path"],
                                        "Size (KB)": round((file_item.get("size") or 0) / 1024, 1),
                                        "Modified": file_item.get("last_modified_datetime"),
                                        "KB Record": file_item.get("staged_document_id") or "",
                                    }
                                    for file_item in onedrive_files
                                ],
                                use_container_width=True,
                                hide_index=True,
                                height=260,
                            )
                        with st.expander("Stage Files for Review", expanded=False):
                            selection_columns = st.columns(3)

                            with selection_columns[0]:
                                if st.button(
                                    "Select All",
                                    use_container_width=True,
                                    key="select_all_onedrive_files_button",
                                ):
                                    st.session_state["selected_onedrive_files_to_stage"] = list(file_options.keys())
                                    st.rerun()

                            with selection_columns[1]:
                                if st.button(
                                    "Select New/Rejected",
                                    use_container_width=True,
                                    key="select_stageable_onedrive_files_button",
                                ):
                                    st.session_state["selected_onedrive_files_to_stage"] = [
                                        label
                                        for label, file_item in file_options.items()
                                        if file_item.get("connector_state", "New") in ["New", "Rejected"]
                                    ]
                                    st.rerun()

                            with selection_columns[2]:
                                if st.button(
                                    "Clear Selection",
                                    use_container_width=True,
                                    key="clear_onedrive_selection_button",
                                ):
                                    st.session_state["selected_onedrive_files_to_stage"] = []
                                    st.rerun()

                            selected_file_labels = st.multiselect(
                                "Select OneDrive files to stage",
                                list(file_options.keys()),
                                key="selected_onedrive_files_to_stage",
                            )

                            if st.button(
                                "Stage Selected Files for Review",
                                use_container_width=True,
                                disabled=(
                                    not selected_file_labels
                                    or bool(st.session_state.get("active_onedrive_stage_job_id"))
                                ),
                                key="submit_onedrive_stage_job_button",
                            ):
                                selected_files = [
                                    file_options[selected_file_label]
                                    for selected_file_label in selected_file_labels
                                ]

                                try:
                                    job = submit_onedrive_stage_job(selected_files)
                                except requests.exceptions.HTTPError as error:
                                    st.error(f"OneDrive staging rejected by backend: {error.response.text}")
                                except requests.exceptions.RequestException as error:
                                    st.error(f"Could not submit OneDrive staging job: {error}")
                                else:
                                    st.session_state["active_onedrive_stage_job_id"] = job["job_id"]
                                    st.session_state["onedrive_stage_message"] = job["message"]
                                    st.session_state["onedrive_stage_status"] = "info"
                                    st.rerun()

                            if st.session_state.get("onedrive_stage_results"):
                                st.dataframe(
                                    st.session_state["onedrive_stage_results"],
                                    use_container_width=True,
                                    hide_index=True,
                                    height=220,
                                )

                    refreshable_file_options = {
                        label: file_item
                        for label, file_item in file_options.items()
                        if file_item.get("connector_state") in ["Pending Index", "Indexed"]
                    }

                    if refreshable_file_options:
                        with st.expander("Refresh OneDrive Sources", expanded=False):
                            if st.session_state.get("onedrive_refresh_message"):
                                refresh_status = st.session_state.get("onedrive_refresh_status", "info")

                                if refresh_status == "success":
                                    st.success(st.session_state["onedrive_refresh_message"])
                                elif refresh_status == "error":
                                    st.error(st.session_state["onedrive_refresh_message"])
                                else:
                                    st.info(st.session_state["onedrive_refresh_message"])

                            refresh_columns = st.columns(3)

                            with refresh_columns[0]:
                                if st.button(
                                    "Select All Refreshable",
                                    use_container_width=True,
                                    key="select_all_onedrive_refresh_files_button",
                                ):
                                    st.session_state["selected_onedrive_files_to_refresh"] = list(
                                        refreshable_file_options.keys()
                                    )
                                    st.rerun()

                            with refresh_columns[1]:
                                if st.button(
                                    "Clear Selection",
                                    use_container_width=True,
                                    key="clear_onedrive_refresh_selection_button",
                                ):
                                    st.session_state["selected_onedrive_files_to_refresh"] = []
                                    st.rerun()

                            with refresh_columns[2]:
                                st.caption(f"{len(refreshable_file_options)} available")

                            selected_refresh_labels = st.multiselect(
                                "Select OneDrive files to refresh",
                                list(refreshable_file_options.keys()),
                                key="selected_onedrive_files_to_refresh",
                            )

                            if st.button(
                                "Refresh Selected OneDrive Files",
                                use_container_width=True,
                                disabled=(
                                    not selected_refresh_labels
                                    or bool(st.session_state.get("active_onedrive_refresh_job_id"))
                                ),
                                key="submit_onedrive_refresh_job_button",
                            ):
                                selected_refresh_files = [
                                    refreshable_file_options[selected_refresh_label]
                                    for selected_refresh_label in selected_refresh_labels
                                ]

                                try:
                                    job = submit_onedrive_refresh_job(selected_refresh_files)
                                except requests.exceptions.HTTPError as error:
                                    st.error(f"OneDrive refresh rejected by backend: {error.response.text}")
                                except requests.exceptions.RequestException as error:
                                    st.error(f"Could not submit OneDrive refresh job: {error}")
                                else:
                                    st.session_state["active_onedrive_refresh_job_id"] = job["job_id"]
                                    st.session_state["onedrive_refresh_message"] = job["message"]
                                    st.session_state["onedrive_refresh_status"] = "info"
                                    st.rerun()

                            if st.session_state.get("onedrive_refresh_results"):
                                st.dataframe(
                                    st.session_state["onedrive_refresh_results"],
                                    use_container_width=True,
                                    hide_index=True,
                                    height=220,
                                )
                                render_pending_index_notice(
                                    [
                                        document
                                        for document in pending_connector_versions
                                        if document.get("source") == "onedrive"
                                    ],
                                    label="OneDrive-updated document",
                                    expander_label="Pending OneDrive Versions",
                                    use_expander=False,
                                )

                            selected_refresh_label = st.selectbox(
                                "Select one OneDrive file to refresh",
                                list(refreshable_file_options.keys()),
                                key="selected_onedrive_file_to_refresh",
                            )

                            if st.button(
                                "Refresh One File",
                                use_container_width=True,
                                key="refresh_selected_onedrive_file_button",
                            ):
                                try:
                                    with st.spinner("Checking OneDrive content and version history..."):
                                        refresh_result = request_onedrive_file_refresh(
                                            refreshable_file_options[selected_refresh_label]
                                        )
                                except requests.exceptions.HTTPError as error:
                                    st.error(f"OneDrive refresh rejected by backend: {error.response.text}")
                                except requests.exceptions.RequestException as error:
                                    st.error(f"Could not refresh OneDrive file: {error}")
                                else:
                                    if refresh_result["status"] == "updated":
                                        st.success(refresh_result["message"])
                                    elif refresh_result["status"] == "no_change":
                                        st.info(refresh_result["message"])
                                    else:
                                        st.warning(refresh_result["message"])

            with onenote_tab:
                with st.container(border=True):
                    render_workflow_header(
                        "OneNote",
                        "Scan the configured notebook scope, stage pages for metadata review, and refresh approved page versions.",
                    )
                    if st.button(
                        "Scan OneNote Notebook",
                        use_container_width=True,
                        key="scan_onenote_pages_button",
                    ):
                        try:
                            scan_result = request_onenote_page_scan()
                        except requests.exceptions.HTTPError as error:
                            st.error(f"OneNote scan rejected by backend: {error.response.text}")
                        except requests.exceptions.RequestException as error:
                            st.error(f"Could not scan OneNote connector: {error}")
                        else:
                            st.session_state["onenote_pages"] = scan_result["pages"]
                            st.success(f"Found {len(scan_result['pages'])} page(s).")

                    onenote_pages = st.session_state.get("onenote_pages", [])

                    if onenote_pages:
                        page_options = {
                            f"{page.get('title') or 'Untitled Page'} - {page['connector_path']}": page
                            for page in onenote_pages
                        }
                        stageable_count = sum(
                            1
                            for page in onenote_pages
                            if page.get("connector_state", "New") in ["New", "Rejected"]
                        )
                        refreshable_count = sum(
                            1
                            for page in onenote_pages
                            if page.get("connector_state") in ["Pending Index", "Indexed"]
                        )
                        connector_metric_columns = st.columns(3)

                        with connector_metric_columns[0]:
                            render_status_card("Discovered", len(onenote_pages))

                        with connector_metric_columns[1]:
                            render_status_card(
                                "Stageable",
                                stageable_count,
                                "attention" if stageable_count else "neutral",
                            )

                        with connector_metric_columns[2]:
                            render_status_card("Refreshable", refreshable_count)

                        with st.expander("Discovered Pages", expanded=False):
                            st.dataframe(
                                [
                                    {
                                        "Title": page["title"] or "Untitled Page",
                                        "State": page.get("connector_state", "New"),
                                        "Notebook": page["notebook_name"],
                                        "Section": page["section_name"],
                                        "Path": page["connector_path"],
                                        "Modified": page.get("last_modified_datetime"),
                                        "KB Record": page.get("staged_document_id") or "",
                                    }
                                    for page in onenote_pages
                                ],
                                use_container_width=True,
                                hide_index=True,
                                height=260,
                            )

                        with st.expander("Stage Pages for Review", expanded=False):
                            selection_columns = st.columns(3)

                            with selection_columns[0]:
                                if st.button(
                                    "Select All Pages",
                                    use_container_width=True,
                                    key="select_all_onenote_pages_button",
                                ):
                                    st.session_state["selected_onenote_pages_to_stage"] = list(page_options.keys())
                                    st.rerun()

                            with selection_columns[1]:
                                if st.button(
                                    "Select New/Rejected",
                                    use_container_width=True,
                                    key="select_stageable_onenote_pages_button",
                                ):
                                    st.session_state["selected_onenote_pages_to_stage"] = [
                                        label
                                        for label, page in page_options.items()
                                        if page.get("connector_state", "New") in ["New", "Rejected"]
                                    ]
                                    st.rerun()

                            with selection_columns[2]:
                                if st.button(
                                    "Clear Selection",
                                    use_container_width=True,
                                    key="clear_onenote_selection_button",
                                ):
                                    st.session_state["selected_onenote_pages_to_stage"] = []
                                    st.rerun()

                            selected_page_labels = st.multiselect(
                                "Select OneNote pages to stage",
                                list(page_options.keys()),
                                key="selected_onenote_pages_to_stage",
                            )

                            if st.session_state.get("onenote_stage_message"):
                                stage_status = st.session_state.get("onenote_stage_status", "info")

                                if stage_status == "success":
                                    st.success(st.session_state["onenote_stage_message"])
                                elif stage_status == "error":
                                    st.error(st.session_state["onenote_stage_message"])
                                else:
                                    st.info(st.session_state["onenote_stage_message"])

                            if st.button(
                                "Stage Selected Pages for Review",
                                use_container_width=True,
                                disabled=(
                                    not selected_page_labels
                                    or bool(st.session_state.get("active_onenote_stage_job_id"))
                                ),
                                key="submit_onenote_stage_job_button",
                            ):
                                selected_pages = [
                                    page_options[selected_page_label]
                                    for selected_page_label in selected_page_labels
                                ]

                                try:
                                    job = submit_onenote_stage_job(selected_pages)
                                except requests.exceptions.HTTPError as error:
                                    st.error(f"OneNote staging rejected by backend: {error.response.text}")
                                except requests.exceptions.RequestException as error:
                                    st.error(f"Could not submit OneNote staging job: {error}")
                                else:
                                    st.session_state["active_onenote_stage_job_id"] = job["job_id"]
                                    st.session_state["onenote_stage_message"] = job["message"]
                                    st.session_state["onenote_stage_status"] = "info"
                                    st.rerun()

                            if st.session_state.get("onenote_stage_results"):
                                st.dataframe(
                                    st.session_state["onenote_stage_results"],
                                    use_container_width=True,
                                    hide_index=True,
                                    height=220,
                                )

                        refreshable_page_options = {
                            label: page
                            for label, page in page_options.items()
                            if page.get("connector_state") in ["Pending Index", "Indexed"]
                        }

                        if refreshable_page_options:
                            with st.expander("Refresh OneNote Pages", expanded=False):
                                if st.session_state.get("onenote_refresh_message"):
                                    refresh_status = st.session_state.get("onenote_refresh_status", "info")

                                    if refresh_status == "success":
                                        st.success(st.session_state["onenote_refresh_message"])
                                    elif refresh_status == "error":
                                        st.error(st.session_state["onenote_refresh_message"])
                                    else:
                                        st.info(st.session_state["onenote_refresh_message"])

                                refresh_columns = st.columns(3)

                                with refresh_columns[0]:
                                    if st.button(
                                        "Select All Refreshable",
                                        use_container_width=True,
                                        key="select_all_onenote_refresh_pages_button",
                                    ):
                                        st.session_state["selected_onenote_pages_to_refresh"] = list(
                                            refreshable_page_options.keys()
                                        )
                                        st.rerun()

                                with refresh_columns[1]:
                                    if st.button(
                                        "Clear Selection",
                                        use_container_width=True,
                                        key="clear_onenote_refresh_selection_button",
                                    ):
                                        st.session_state["selected_onenote_pages_to_refresh"] = []
                                        st.rerun()

                                with refresh_columns[2]:
                                    st.caption(f"{len(refreshable_page_options)} available")

                                selected_refresh_labels = st.multiselect(
                                    "Select OneNote pages to refresh",
                                    list(refreshable_page_options.keys()),
                                    key="selected_onenote_pages_to_refresh",
                                )

                                if st.button(
                                    "Refresh Selected OneNote Pages",
                                    use_container_width=True,
                                    disabled=(
                                        not selected_refresh_labels
                                        or bool(st.session_state.get("active_onenote_refresh_job_id"))
                                    ),
                                    key="submit_onenote_refresh_job_button",
                                ):
                                    selected_refresh_pages = [
                                        refreshable_page_options[selected_refresh_label]
                                        for selected_refresh_label in selected_refresh_labels
                                    ]

                                    try:
                                        job = submit_onenote_refresh_job(selected_refresh_pages)
                                    except requests.exceptions.HTTPError as error:
                                        st.error(f"OneNote refresh rejected by backend: {error.response.text}")
                                    except requests.exceptions.RequestException as error:
                                        st.error(f"Could not submit OneNote refresh job: {error}")
                                    else:
                                        st.session_state["active_onenote_refresh_job_id"] = job["job_id"]
                                        st.session_state["onenote_refresh_message"] = job["message"]
                                        st.session_state["onenote_refresh_status"] = "info"
                                        st.rerun()

                                if st.session_state.get("onenote_refresh_results"):
                                    st.dataframe(
                                        st.session_state["onenote_refresh_results"],
                                        use_container_width=True,
                                        hide_index=True,
                                        height=220,
                                    )
                                    render_pending_index_notice(
                                        [
                                            document
                                            for document in pending_connector_versions
                                            if document.get("source") == "onenote"
                                        ],
                                        label="OneNote-updated document",
                                        expander_label="Pending OneNote Versions",
                                        use_expander=False,
                                    )

                                selected_refresh_label = st.selectbox(
                                    "Select one OneNote page to refresh",
                                    list(refreshable_page_options.keys()),
                                    key="selected_onenote_page_to_refresh",
                                )

                                if st.button(
                                    "Refresh One Page",
                                    use_container_width=True,
                                    key="refresh_selected_onenote_page_button",
                                ):
                                    try:
                                        with st.spinner("Checking OneNote content and version history..."):
                                            refresh_result = request_onenote_page_refresh(
                                                refreshable_page_options[selected_refresh_label]
                                            )
                                    except requests.exceptions.HTTPError as error:
                                        st.error(f"OneNote refresh rejected by backend: {error.response.text}")
                                    except requests.exceptions.RequestException as error:
                                        st.error(f"Could not refresh OneNote page: {error}")
                                    else:
                                        if refresh_result["status"] == "updated":
                                            st.success(refresh_result["message"])
                                        elif refresh_result["status"] == "no_change":
                                            st.info(refresh_result["message"])
                                        else:
                                            st.warning(refresh_result["message"])

                    else:
                        st.caption("No OneNote pages loaded from the configured notebook scope.")

        with review_tab:
            with st.container(border=True):
                render_workflow_header(
                    "Review Queue",
                    "Review connector-staged documents, confirm metadata, and approve or reject them before indexing.",
                )

                reviewable_documents = [
                    document
                    for document in pending_review_documents
                    if (
                        st.session_state["role"] == "System Admin"
                        or document["department"] == st.session_state["department"]
                    )
                ]

                if not reviewable_documents:
                    st.info("No connector documents are waiting for review.")
                else:
                    review_rows = [
                        {
                            "Document ID": document["document_id"],
                            "Title": document["title"],
                            "Source": document["source"],
                            "Department": document["department"],
                            "Uploaded At": document.get("uploaded_at", ""),
                        }
                        for document in reviewable_documents
                    ]

                    st.dataframe(
                        review_rows,
                        use_container_width=True,
                        hide_index=True,
                        height=220,
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
                    with st.expander("Reject Staged Document", expanded=False):
                        st.warning(
                            "Rejecting removes this connector import from pending review. "
                            "The record is kept inactive for audit history and will not be indexed."
                        )

                        confirm_reject = st.checkbox(
                            f"I understand this will reject {selected_review_document['title']}.",
                            key=f"confirm_reject_{selected_review_document['document_id']}",
                        )

                        if st.button(
                            "Reject Staged Document",
                            key=f"reject_staged_{selected_review_document['document_id']}",
                            disabled=not confirm_reject,
                        ):
                            try:
                                rejection_result = request_staged_document_rejection(
                                    selected_review_document["document_id"]
                                )
                            except requests.exceptions.HTTPError as error:
                                st.error(f"Rejection rejected by backend: {error.response.text}")
                            except requests.exceptions.RequestException as error:
                                st.error(f"Could not reject staged document: {error}")
                            else:
                                st.success(rejection_result["message"])
                                st.rerun()

        if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
            with index_tab:
                with st.container(border=True):
                    render_workflow_header(
                        "Index Sync",
                        "Apply pending document updates to the active search index or run a full rebuild when backend settings change.",
                    )
                    render_pending_index_notice(
                        pending_index_documents,
                        label="pending document",
                        expander_label="Pending Documents",
                        show_index_sync_hint=False,
                    )
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
                        confirm_full_rebuild = st.checkbox(
                            "Confirm full rebuild",
                            key="confirm_full_rebuild",
                            help="Full rebuild recreates the active search index from current active metadata.",
                        )

                        if st.button(
                            "Full Rebuild",
                            use_container_width=True,
                            disabled=(
                                not confirm_full_rebuild
                                or bool(st.session_state.get("active_reindex_job_id"))
                            ),
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
        visual_status_options = sorted(
            {document["visual_extraction_status"] for document in visible_documents}
        )
        index_status_options = [
            FILTER_ALL,
            "Pending Index",
            "Indexed",
        ]

        filtered_documents = visible_documents

        if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
            pending_index_count = sum(
                1 for document in visible_documents
                if get_index_status_label(document) == "Pending Index"
            )

            if pending_index_count:
                if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
                    st.warning(
                        f"{pending_index_count} visible document(s) need index update before chat uses the latest content."
                    )
                else:
                    st.warning(
                        f"{pending_index_count} visible document(s) are waiting for System Admin index update."
                    )

        if st.session_state["role"] == GENERAL_EMPLOYEE_ROLE:
            st.subheader("Knowledge Library")
        else:
            st.subheader("Knowledge Library")

        with st.container(border=True):

            filter_columns = st.columns(5 if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE else 4)

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

            selected_visual_status = FILTER_ALL

            with filter_columns[3]:
                selected_visual_status = st.selectbox(
                    "Visual Status",
                    [FILTER_ALL] + visual_status_options,
                )

            selected_index_status = FILTER_ALL
            
            if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
                with filter_columns[4]:
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

        if selected_visual_status != FILTER_ALL:
            filtered_documents = [
                document
                for document in filtered_documents
                if document["visual_extraction_status"] == selected_visual_status
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
            with st.expander("Selected Document", expanded=True):
                selected_title = st.selectbox(
                    "Document",
                    [document["title"] for document in filtered_documents],
                    key="selected_library_document",
                )

                selected_document = next(
                    document for document in filtered_documents
                    if document["title"] == selected_title
                )

                overview_tab, storage_tab, edit_tab, archive_tab = st.tabs([
                    "Overview",
                    "Storage",
                    "Edit Metadata",
                    "Archive",
                ])

                with overview_tab:
                    overview_columns = st.columns(3)

                    with overview_columns[0]:
                        st.markdown("**Identity**")
                        st.write(f"**ID:** {selected_document['document_id']}")
                        st.write(f"**File:** {selected_document['filename']}")
                        st.write(f"**Type:** {selected_document['file_type']}")
                        if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
                            st.write(f"**Version:** {get_version_label(selected_document)}")

                    with overview_columns[1]:
                        st.markdown("**Governance**")
                        st.write(f"**Department:** {selected_document['department']}")
                        st.write(f"**Category:** {selected_document['category']}")
                        st.write(f"**Tags:** {', '.join(selected_document['tags'])}")
                        if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
                            st.write(f"**Status:** {get_index_status_label(selected_document)}")

                    with overview_columns[2]:
                        st.markdown("**Access**")
                        if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
                            st.write(f"**Roles:** {', '.join(selected_document['allowed_roles'])}")
                            st.write(f"**Departments:** {', '.join(selected_document['allowed_departments'])}")
                            st.write(f"**Index marker:** {selected_document['chunk_id']}")
                        st.write(f"**Visuals:** {selected_document['visual_extraction_status']}")

                with storage_tab:
                    st.write(f"**Storage backend:** {selected_document.get('storage_backend', 'local')}")
                    st.code(
                        selected_document.get(
                            "storage_uri",
                            f"data/simulated/{selected_document['filename']}",
                        ),
                        language=None,
                    )
                    st.write(
                        f"**Uploaded by:** {selected_document['uploaded_by']} "
                        f"at {selected_document['uploaded_at']}"
                    )
                    if st.session_state["role"] != GENERAL_EMPLOYEE_ROLE:
                        st.write(
                            f"**Source document ID:** "
                            f"{selected_document.get('source_document_id')}"
                        )

                with edit_tab:
                    if st.session_state["role"] not in [SYSTEM_ADMIN_ROLE, PROJECT_MANAGER_ROLE]:
                        st.info("Metadata editing is not available for your role.")
                    else:
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

                            submitted_metadata_update = st.form_submit_button(
                                "Save Metadata",
                                type="primary",
                            )

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

                with archive_tab:
                    if st.session_state["role"] not in [SYSTEM_ADMIN_ROLE, PROJECT_MANAGER_ROLE]:
                        st.info("Archive and restore actions are not available for your role.")

                    else:
                        can_archive_selected_document = (
                            st.session_state["role"] == SYSTEM_ADMIN_ROLE
                            or selected_document["department"] == st.session_state["department"]
                        )

                        if not can_archive_selected_document:
                            st.info("Project Manager can only archive own-department documents.")
                        else:
                            st.warning(
                                "Archiving removes this document from active retrieval "
                                "and deletes its vector/index records from the configured backend."
                            )

                            confirm_archive = st.checkbox(
                                f"I understand this will archive {selected_document['title']}.",
                                key=f"confirm_archive_{selected_document['document_id']}",
                            )

                            if st.button(
                                "Archive Selected Document",
                                key=f"archive_document_{selected_document['document_id']}",
                                disabled=(
                                    not confirm_archive
                                    or bool(st.session_state.get("active_document_lifecycle_job_id"))
                                ),
                            ):
                                try:
                                    job = request_document_archive(
                                        selected_document["document_id"]
                                    )
                                except requests.exceptions.HTTPError as error:
                                    st.session_state["document_lifecycle_message"] = (
                                        f"Archive rejected by backend: {error.response.text}"
                                    )
                                    st.session_state["document_lifecycle_status"] = "error"
                                except requests.exceptions.RequestException as error:
                                    st.session_state["document_lifecycle_message"] = (
                                        f"Could not submit archive job: {error}"
                                    )
                                    st.session_state["document_lifecycle_status"] = "error"
                                else:
                                    st.session_state["active_document_lifecycle_job_id"] = job["job_id"]
                                    st.session_state["document_lifecycle_message"] = job["message"]
                                    st.session_state["document_lifecycle_status"] = "info"

                                st.rerun()

    if st.session_state["role"] in [SYSTEM_ADMIN_ROLE, PROJECT_MANAGER_ROLE]:
        with st.expander("Archived Documents", expanded=False):
            st.caption(
                "Only manually archived documents are shown here. "
                "Old versions replaced by newer documents are kept as history and cannot be restored."
            )

            if not restorable_archived_documents:
                st.info("No manually archived documents can be restored.")
            else:
                archived_rows = [
                    {
                        "Document": document["title"],
                        "Document ID": document["document_id"],
                        "Department": document["department"],
                        "Source": document["source"],
                        "Archived At": document.get("archived_at") or "",
                    }
                    for document in restorable_archived_documents
                ]

                st.dataframe(
                    archived_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                archived_options = {
                    f"{document['title']} ({document['document_id']})": document
                    for document in restorable_archived_documents
                }

                selected_archived_label = st.selectbox(
                    "Select archived document to restore",
                    list(archived_options.keys()),
                    key="selected_archived_document_to_restore",
                )

                selected_archived_document = archived_options[selected_archived_label]

                confirm_restore = st.checkbox(
                    f"I understand this will restore {selected_archived_document['title']}.",
                    key=f"confirm_restore_{selected_archived_document['document_id']}",
                )

                if st.button(
                    "Restore Archived Document",
                    key=f"restore_document_{selected_archived_document['document_id']}",
                    disabled=(
                        not confirm_restore
                        or bool(st.session_state.get("active_document_lifecycle_job_id"))
                    ),
                ):
                    try:
                        job = request_document_unarchive(
                            selected_archived_document["document_id"]
                        )
                    except requests.exceptions.HTTPError as error:
                        st.session_state["document_lifecycle_message"] = (
                            f"Restore rejected by backend: {error.response.text}"
                        )
                        st.session_state["document_lifecycle_status"] = "error"
                    except requests.exceptions.RequestException as error:
                        st.session_state["document_lifecycle_message"] = (
                            f"Could not submit restore job: {error}"
                        )
                        st.session_state["document_lifecycle_status"] = "error"
                    else:
                        st.session_state["active_document_lifecycle_job_id"] = job["job_id"]
                        st.session_state["document_lifecycle_message"] = job["message"]
                        st.session_state["document_lifecycle_status"] = "info"

                    st.rerun()

    st.stop()


if selected_page == "Chat":
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    if "chat_session_id" not in st.session_state:
        st.session_state["chat_session_id"] = None

    if "chat_memory_enabled" not in st.session_state:
        st.session_state["chat_memory_enabled"] = True

    active_session_label = (
        "Current conversation"
        if not st.session_state["chat_session_id"]
        else f"Session {st.session_state['chat_session_id'][:8]}"
    )
    memory_label = "Memory on" if st.session_state["chat_memory_enabled"] else "Memory off"

    try:
        persisted_sessions = get_backend_chat_sessions()
    except requests.exceptions.RequestException:
        persisted_sessions = []

    session_labels = {
        (
            f"{session['title']} "
            f"({session['updated_at']})"
        ): session["session_id"]
        for session in persisted_sessions
    }

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

    if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
        department_filter = FILTER_ALL
        file_type_filter = FILTER_ALL
    elif st.session_state["role"] == PROJECT_MANAGER_ROLE:
        department_filter = None
        file_type_filter = FILTER_ALL
    else:
        department_filter = None
        file_type_filter = FILTER_ALL

    chat_is_processing = st.session_state.get("chat_is_processing", False)
    workspace_columns = st.columns([1.1, 3.9], gap="medium")

    with workspace_columns[0]:
        st.markdown('<span class="chat-history-panel-marker"></span>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size:0.95rem; font-weight:850; color:#101828; margin-bottom:0.35rem;">
                    Conversations
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "New Chat",
                use_container_width=True,
                disabled=chat_is_processing,
            ):
                st.session_state["chat_session_id"] = None
                st.session_state["chat_messages"] = []
                st.session_state.pop("active_chat_job_id", None)
                st.session_state["chat_is_processing"] = False
                st.rerun()

            st.toggle(
                "Use Memory",
                key="chat_memory_enabled",
                help=(
                    "When enabled, the backend uses the last 6 messages to rewrite "
                    "follow-up questions before normal ACL-filtered retrieval."
                ),
                disabled=chat_is_processing,
            )

            st.markdown(
                f"""
                <div style="
                    margin:0.35rem 0 0.55rem 0;
                    padding:0.55rem;
                    background:#f9fafb;
                    border:1px solid #eaecf0;
                    border-radius:8px;
                ">
                    <div style="font-size:0.72rem; color:#667085; font-weight:750;">Active</div>
                    <div style="font-size:0.86rem; color:#101828; font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                        {escape(active_session_label)}
                    </div>
                    <div style="font-size:0.76rem; color:#667085; margin-top:0.2rem;">
                        {escape(memory_label)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not persisted_sessions:
                st.caption("No saved conversations yet.")
            else:
                selected_session_label = st.selectbox(
                    "Recent",
                    ["Select conversation"] + list(session_labels.keys()),
                    label_visibility="collapsed",
                    disabled=chat_is_processing,
                )
                selected_session_id = session_labels.get(selected_session_label)

                if st.button(
                    "Open",
                    use_container_width=True,
                    disabled=selected_session_id is None or chat_is_processing,
                ):
                    try:
                        persisted_messages = get_backend_chat_session_messages(selected_session_id)
                    except requests.exceptions.RequestException as error:
                        st.warning(f"Could not load conversation: {error}")
                    else:
                        st.session_state["chat_session_id"] = selected_session_id
                        st.session_state["chat_messages"] = [
                            convert_persisted_message_to_chat_message(message)
                            for message in persisted_messages
                        ]
                        st.rerun()

            if st.button("Clear Current", use_container_width=True, disabled=chat_is_processing):
                st.session_state["chat_messages"] = []
                st.session_state["chat_session_id"] = None
                st.rerun()

    with workspace_columns[1]:
        st.markdown('<span class="chat-main-panel-marker"></span>', unsafe_allow_html=True)
        with st.container(border=True):
            chat_header_columns = st.columns([2.7, 1, 1])

            with chat_header_columns[0]:
                st.markdown(
                    f"""
                    <div style="padding:0.05rem 0 0.35rem 0;">
                        <div style="display:flex; align-items:center; gap:0.45rem; flex-wrap:wrap;">
                            <span style="font-size:1.35rem; font-weight:875; color:#101828;">Copilot Chat</span>
                            <span class="status-pill" style="
                                font-size:0.74rem;
                                padding:0.16rem 0.42rem;
                                color:#344054;
                                background:#ffffff;
                                border-color:#d0d5dd;
                            ">{escape(st.session_state["role"])} / {escape(st.session_state["department"])}</span>
                        </div>
                        <div style="font-size:0.82rem; color:#667085; margin-top:0.16rem;">
                            Ask from your permitted knowledge scope. Answers refresh retrieval and citations each time.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
                with chat_header_columns[1]:
                    department_filter = st.selectbox(
                        "Department",
                        [FILTER_ALL] + available_departments,
                        label_visibility="collapsed",
                    )

                with chat_header_columns[2]:
                    file_type_filter = st.selectbox(
                        "File Type",
                        [FILTER_ALL] + available_file_types,
                        label_visibility="collapsed",
                    )

                filter_status = (
                    f"Scope: Department = {department_filter}, "
                    f"File Type = {file_type_filter}"
                )

            elif st.session_state["role"] == PROJECT_MANAGER_ROLE:
                with chat_header_columns[1]:
                    st.text_input(
                        "Department",
                        value=f"{st.session_state['department']} + shared",
                        disabled=True,
                        label_visibility="collapsed",
                    )

                with chat_header_columns[2]:
                    file_type_filter = st.selectbox(
                        "File Type",
                        [FILTER_ALL] + available_file_types,
                        label_visibility="collapsed",
                    )

                filter_status = (
                    f"Scope: {st.session_state['department']} + shared ACL docs, "
                    f"File Type = {file_type_filter}"
                )

            else:
                with chat_header_columns[1]:
                    st.text_input(
                        "Department",
                        value=st.session_state["department"],
                        disabled=True,
                        label_visibility="collapsed",
                    )

                with chat_header_columns[2]:
                    st.text_input(
                        "File Type",
                        value="All allowed",
                        disabled=True,
                        label_visibility="collapsed",
                    )

                filter_status = "Scope: own department plus ACL-shared documents"

            st.markdown(
                f"""
                <div style="
                    border-top:1px solid #eaecf0;
                    margin:0.2rem 0 0.65rem 0;
                    padding-top:0.55rem;
                    color:#667085;
                    font-size:0.82rem;
                ">{escape(filter_status)}</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<span class="chat-message-panel-marker"></span>', unsafe_allow_html=True)
            chat_container = st.container(height=560, border=True)

            with chat_container:
                if not st.session_state["chat_messages"]:
                    st.markdown(
                        f"<div style='text-align: center; color: #667085; margin-top: 2.5rem;'>"
                        f"<div style='font-weight:750; color:#101828; margin-bottom:0.25rem;'>"
                        f"Hello {escape(st.session_state['user'])}</div>"
                        f"<div>Ask a grounded question from your permitted knowledge scope.</div>"
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

                        if message.get("sources") or message.get("context"):
                            meta_col1, meta_col2 = st.columns(2)
                            if message.get("sources"):
                                source_label = (
                                    "Sources Checked"
                                    if message.get("status") == "not_found"
                                    else "Sources"
                                )

                                with meta_col1:
                                    with st.expander(source_label):
                                        for source in message["sources"]:
                                            st.code(source, language=None)

                            if message.get("context"):
                                with meta_col2:
                                    with st.expander("Query Context"):
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

                if st.session_state.get("active_chat_job_id"):
                    with st.chat_message("assistant"):
                        st.markdown(
                            """
                            <div class="typing-indicator">
                                <span class="typing-dot"></span>
                                <span class="typing-dot"></span>
                                <span class="typing-dot"></span>
                                <span>Drafting an answer from permitted knowledge-base content.</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

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

            example_prompts = ROLE_AWARE_CHAT_PROMPTS[st.session_state["role"]]

            with st.expander("Suggested Questions", expanded=False):
                total_cols = len(example_prompts) + (1 if st.session_state["role"] == GENERAL_EMPLOYEE_ROLE else 0)
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
                    with example_columns[-1]:
                        st.button(
                            "Restricted IT Policy",
                            key="acl_demo_prompt",
                            on_click=select_example_chat_prompt,
                            args=("What are the password policy requirements?",),
                            use_container_width=True,
                            disabled=chat_is_processing,
                        )

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

    st.stop()


if selected_page == "Settings":
    st.header("System Settings")
    st.caption("Admin-only backend mode, retrieval, and guardrail configuration.")

    try:
        settings_response = request_admin_settings()
        current_settings = settings_response["settings"]
        runtime_info = settings_response.get("runtime_info", {})
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
        mode_columns = st.columns(4)
        mode_columns[0].metric("Storage", current_settings["storage_backend"])
        mode_columns[1].metric("Vector", current_settings["vector_backend"])
        mode_columns[2].metric("Embedding", current_settings["embedding_backend"])
        mode_columns[3].metric("LLM", current_settings["llm_backend"])
        st.caption(
            "Active generation deployment: "
            f"{runtime_info.get('llm_deployment', 'Not configured')}"
        )

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
        * **SharePoint** is not included in this build because the current account does not have enterprise Microsoft 365 SharePoint site access. The future path requires a company tenant, test site/library, and Graph permissions.
        """)
