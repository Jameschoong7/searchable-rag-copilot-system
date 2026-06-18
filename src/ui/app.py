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
    append_document_metadata,
    create_new_document_version,
    generate_document_id,
    load_document_metadata,
    metadata_exists_for_filename,
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
INDEX_UPDATES_URL = f"{API_BASE_URL}/admin/index-updates"
API_URL = f"{API_BASE_URL}/query"
API_HEALTH_URL = f"{API_BASE_URL}/health"
METADATA_UPDATE_VALIDATE_URL = f"{API_BASE_URL}/admin/validate-metadata-update"
QUERY_LOG_DB_PATH = PROJECT_ROOT / "data/logs/query_logs.db"
EVALUATION_RESULTS_PATH = PROJECT_ROOT / "data/evaluation/retrieval_eval_results.json"
INDEX_BENCHMARK_RESULTS_PATH = PROJECT_ROOT / "data/evaluation/index_benchmark_results.json"
INDEX_BENCHMARK_HISTORY_PATH = PROJECT_ROOT / "data/evaluation/index_benchmark_history.json"


def request_backend_reindex() -> dict:
    """Ask the FastAPI backend to rebuild the local vector index."""
    response = requests.post(
        REINDEX_URL,
        json={
            "role": st.session_state["role"],
        },
        timeout=300,
    )

    response.raise_for_status()
    return response.json()


def request_pending_index_update() -> dict:
    """Ask FastAPI to incrementally index pending document updates."""
    response = requests.post(
        INDEX_UPDATES_URL,
        json={
            "role": st.session_state["role"],
        },
        timeout=300,
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


def write_query_log(
        question: str,
        department_filter: str | None,
        file_type_filter: str | None,
        status: str,
        sources: list[str],
        latency_seconds: float,
) -> None:
    """Insert one structured chat query event into the local SQLite log."""
    initialise_query_log_database()

    with sqlite3.connect(QUERY_LOG_DB_PATH) as connection:
        connection.execute(
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
            LIMIT 10
            """
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
            "permission_blocks": summary_row[2] or 0,
            "unresolved_queries": summary_row[3] or 0,
            "recent_queries": recent_rows,
            "daily_latency_rows": daily_latency_rows,
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


def generate_version_document_id(previous_document: dict, next_version_number: int) -> str:
    """Generate a readable document ID for a new version of an existing document."""
    source_document_id = previous_document.get(
        "source_document_id",
        previous_document["document_id"],
    )

    return f"{source_document_id}-V{next_version_number}"


def build_versioned_filename(
    previous_document: dict,
    uploaded_filename: str,
    next_version_number: int,
) -> str:
    """Build a unique stored filename for a replacement document version."""
    source_document_id = previous_document.get(
        "source_document_id",
        previous_document["document_id"],
    )

    safe_uploaded_filename = normalise_uploaded_filename(uploaded_filename)

    return f"{source_document_id}_v{next_version_number}_{safe_uploaded_filename}"


def save_uploaded_file_as(uploaded_file, stored_filename: str) -> str:
    """Save an uploaded file using a caller-provided unique stored filename."""
    filename = normalise_uploaded_filename(stored_filename)
    file_path = Path("data/simulated") / filename

    file_path.write_bytes(uploaded_file.getvalue())

    return filename


def get_uploaded_file_type(filename: str) -> str:
    """Return the metadata file type for a supported uploaded file."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        return "TXT"

    if suffix == ".pdf":
        return "PDF"

    if suffix == ".docx":
        return "DOCX"

    return "UNKNOWN"


def get_visual_extraction_status(file_type: str) -> str:
    """Return the local extraction status label for the uploaded file type."""
    if file_type == "PDF":
        return "PDF text extraction"
    
    if file_type == "DOCX":
        return "Word text extraction"

    return "Text only"


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


def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded TXT file into the simulated data folder."""
    filename = normalise_uploaded_filename(uploaded_file.name)
    file_path = Path("data/simulated") / filename

    file_path.write_bytes(uploaded_file.getvalue())

    return filename


def infer_title_from_uploaded_file(uploaded_file) -> str:
    """Infer a default document title from the uploaded filename."""
    return Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ").title()


st.set_page_config(
    page_title="Searchable RAG Copilot",
    layout="wide",
)


if not is_logged_in():
    st.title("Searchable RAG Copilot")
    st.caption("Demo sign-in for the standalone Admin Web Portal.")
    
    with st.container(border=True):
        username = st.text_input("Username", value="admin_jc")
        password = st.text_input("Password", value="password123", type="password")

        if st.button("Sign In", type="primary"):
            if login_user(username.strip(), password):
                st.rerun()
            else:
                st.error("Invalid demo username or password.")

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
                "Top-K Accuracy (K=5)",
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
                "Simulated KB records",
            )

        st.caption(
            "TTFA is calculated from local chat logs. Top-K Accuracy and Miss Rate come "
            "from the latest labelled retrieval evaluation run."
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

    st.subheader("Incremental Index Health")

    if index_benchmark_results:
        after_snapshot = index_benchmark_results.get("after", index_benchmark_results)
        benchmark_type = index_benchmark_results.get("benchmark_type", "snapshot")

        active_vectors = after_snapshot["chroma_vector_count"]
        active_records = after_snapshot["active_metadata_records"]
        physical_files = after_snapshot["simulated_source_files"]
        db_size_mb = after_snapshot["chroma_db_size_mb"]
        archived_file_count = max(physical_files - active_records, 0)

        if benchmark_type == "batch_incremental_update":
            changed_document_count = index_benchmark_results["changed_document_count"]
            chunks_refreshed = index_benchmark_results["total_chunks_indexed"]
            deleted_vectors = index_benchmark_results["total_deleted_vectors"]
            avoided_chunks = index_benchmark_results["estimated_unchanged_chunks_avoided"]
            elapsed_seconds = index_benchmark_results["elapsed_seconds"]
            before_active_vectors = index_benchmark_results["before"]["chroma_vector_count"]

            document_label = (
                "document"
                if changed_document_count == 1
                else "documents"
            )

            st.success(
                f"Latest incremental update refreshed {changed_document_count} changed "
                f"{document_label} in {elapsed_seconds}s. {chunks_refreshed} chunks were "
                f"re-indexed and {avoided_chunks} unchanged chunks were avoided."
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
                    "Chunks Refreshed",
                    chunks_refreshed,
                    f"{avoided_chunks} unchanged avoided",
                )

            with metric_columns[2]:
                st.metric(
                    "Vectors Replaced",
                    deleted_vectors,
                    "Removed before re-index",
                )

            with metric_columns[3]:
                st.metric(
                    "Active Index",
                    f"{active_vectors} vectors",
                    f"{db_size_mb} MB",
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
                time_saved_seconds = max(
                    round(full_rebuild_baseline_seconds - elapsed_seconds, 3),
                    0,
                )
            else:
                full_rebuild_baseline_seconds = None
                full_rebuild_baseline_chunks = None
                time_saved_seconds = None

            with st.expander("Update Efficiency Details", expanded=False):
                efficiency_columns = st.columns(2)

                with efficiency_columns[0]:
                    st.metric(
                        "Work Avoided",
                        f"{avoided_percent:.1f}%",
                        f"{avoided_chunks} chunks skipped",
                    )

                with efficiency_columns[1]:
                    if time_saved_seconds is None:
                        st.metric(
                            "Time Saved",
                            "No baseline",
                            "Run full rebuild first",
                        )
                    else:
                        st.metric(
                            "Time Saved",
                            f"{time_saved_seconds}s",
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
                st.metric("Active Index", f"{active_vectors} vectors", f"{db_size_mb} MB")

        else:
            st.info("Latest result is an index snapshot.")

            metric_columns = st.columns(3)

            with metric_columns[0]:
                st.metric("Active Records", active_records)

            with metric_columns[1]:
                st.metric("Active Index", f"{active_vectors} vectors")

            with metric_columns[2]:
                st.metric("Vector DB Size", f"{db_size_mb} MB")

        if archived_file_count:
            st.warning(
                f"{archived_file_count} archived source file(s) remain on disk for audit, "
                "but active-aware indexing excludes archived versions from Chroma."
            )
        else:
            st.caption("Physical source files and active metadata records are aligned.")

        with st.expander("Technical index benchmark details", expanded=False):
            detail_rows = [
                {"Metric": "Benchmark Type", "Value": benchmark_type},
                {"Metric": "Active Metadata Records", "Value": active_records},
                {"Metric": "Physical Source Files", "Value": physical_files},
                {"Metric": "Archived Physical Files", "Value": archived_file_count},
                {"Metric": "Chroma Vector Count", "Value": active_vectors},
                {"Metric": "Chroma DB Size MB", "Value": db_size_mb},
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
            "**Top-K Accuracy checks whether the expected source document appears "
            "within the top 5 retrieved chunks.**"
        )
        st.caption(
            "Current Top-K Accuracy and Miss Rate use the latest local labelled "
            "retrieval evaluation result. Prototype benchmark values should be treated "
            "as presentation examples only."
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

    with st.expander("View Retrieval Miss Log & Context", expanded=False):
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

    with st.expander("Recent Logged Queries", expanded=False):
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
                        "Uploads a TXT file into data/simulated and appends trusted metadata. "
                        "Rebuild the vector index after upload before searching the new document."
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
                                documents = load_document_metadata()
                                filename = normalise_uploaded_filename(uploaded_file.name)
                                file_type = get_uploaded_file_type(filename)

                                if file_type == "UNKNOWN":
                                    st.error("Only TXT, PDF, and DOCX uploads are supported in the current local prototype.")
                                    st.stop()

                                if metadata_exists_for_filename(filename):
                                    st.error(
                                        "Metadata already exists for this filename. Rename the file or remove the existing metadata record first."
                                    )
                                    st.stop()

                                try:
                                    approved_metadata = request_upload_validation(
                                        document_department=department,
                                        allowed_roles=allowed_roles,
                                        allowed_departments=allowed_departments,
                                    )
                                except requests.exceptions.HTTPError as error:
                                    st.error(f"Upload rejected by backend: {error.response.text}")
                                    st.stop()
                                except requests.exceptions.RequestException as error:
                                    st.error(f"Could not validate upload metadata: {error}")
                                    st.stop()

                                filename = save_uploaded_file(uploaded_file)

                                new_document = {
                                    "document_id": generate_document_id(documents),
                                    "title": title.strip(),
                                    "filename": filename,
                                    "file_type": file_type,
                                    "source": "Manual Upload",
                                    "department": approved_metadata["document_department"],
                                    "category": category.strip() or "General",
                                    "tags": [
                                        tag.strip()
                                        for tag in tags_text.split(",")
                                        if tag.strip()
                                    ],
                                    "allowed_roles": approved_metadata["allowed_roles"],
                                    "allowed_departments": approved_metadata["allowed_departments"],
                                    "uploaded_by": st.session_state["user"],
                                    "uploaded_at": datetime.now().isoformat(timespec="minutes"),
                                    "page_number": None,
                                    "chunk_id": "pending_index",
                                    "visual_extraction_status": get_visual_extraction_status(file_type),
                                }

                                append_document_metadata(new_document)

                                st.session_state["upload_message"] = (
                                    f"Saved {filename} and appended metadata record "
                                    f"{new_document['document_id']}. Rebuild ChromaDB before searching it."
                                )

                                st.session_state["upload_form_version"] += 1
                                st.rerun()

                with new_version_tab:
                    st.markdown("**Upload Replacement As New Version**")
                    st.caption(
                        "Select an existing active document, then upload its replacement. "
                        "The previous metadata record will be archived and the new version will be marked pending index."
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
                                    stored_filename = build_versioned_filename(
                                        selected_version_document,
                                        uploaded_version_file.name,
                                        next_version_number,
                                    )

                                    file_type = get_uploaded_file_type(stored_filename)

                                    if file_type == "UNKNOWN":
                                        st.error("Only TXT, PDF, and DOCX uploads are supported.")
                                        st.stop()

                                    if metadata_exists_for_filename(stored_filename):
                                        st.error(
                                            "A stored file for this version already exists. "
                                            "Please choose a different replacement file or check existing metadata."
                                        )
                                        st.stop()

                                    saved_filename = save_uploaded_file_as(
                                        uploaded_version_file,
                                        stored_filename,
                                    )

                                    new_version_document = selected_version_document.copy()
                                    new_version_document.update(
                                        {
                                            "document_id": generate_version_document_id(
                                                selected_version_document,
                                                next_version_number,
                                            ),
                                            "filename": saved_filename,
                                            "file_type": file_type,
                                            "uploaded_by": st.session_state["user"],
                                            "uploaded_at": datetime.now().isoformat(timespec="minutes"),
                                            "page_number": None,
                                            "chunk_id": "pending_index",
                                            "visual_extraction_status": get_visual_extraction_status(file_type),
                                            "content_hash": None,
                                        }
                                    )

                                    create_new_document_version(
                                        previous_document_id=selected_version_document["document_id"],
                                        new_document=new_version_document,
                                        archived_at=datetime.now().isoformat(timespec="minutes"),
                                    )

                                    st.session_state["upload_message"] = (
                                        f"Created {selected_version_document['title']} "
                                        f"v{next_version_number}. Run Incremental Index Update "
                                        "to remove old vectors and index the new version."
                                    )
                                    st.rerun()

        if st.session_state["role"] == SYSTEM_ADMIN_ROLE:
            with st.container(border=True):
                st.markdown("**2. Vector Index Sync**")
                st.caption(
                    "Use incremental update for pending document changes. Use full rebuild "
                    "when you want to reconstruct the active index from scratch."
                )

                index_action_columns = st.columns(2)

                with index_action_columns[0]:
                    if st.button("Run Incremental Index Update", use_container_width=True):
                        with st.spinner("Indexing pending document updates..."):
                            try:
                                index_update_result = request_pending_index_update()
                            except Exception as error:
                                st.error(f"Incremental index update failed: {error}")
                            else:
                                if index_update_result["status"] == "no_pending_documents":
                                    st.info(index_update_result["message"])
                                else:
                                    st.success(index_update_result["message"])

                with index_action_columns[1]:
                    if st.button("Rebuild Full Active Index", use_container_width=True):
                        with st.spinner("Rebuilding local ChromaDB index..."):
                            try:
                                rebuild_result = request_backend_reindex()
                                rebuild_message = rebuild_result["message"]
                            except Exception as error:
                                st.error(f"Index rebuild failed: {error}")
                            else:
                                st.success(rebuild_message)

        if st.session_state["upload_message"]:
            st.warning(st.session_state["upload_message"])

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
                    st.warning(
                        f"{pending_index_count} document(s) need indexing before chat can use the latest content."
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
                    "Department": document["department"],
                    "Category": document["category"],
                    "Allowed Access": ", ".join(document["allowed_roles"]),
                    "Visuals": document["visual_extraction_status"],
                }
            else:
                row = {
                    "Document": document["title"],
                    "Source": document["source"],
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
                                    st.stop()
                                except requests.exceptions.RequestException as error:
                                    st.error(f"Could not validate metadata update: {error}")
                                    st.stop()

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
        if st.button("🧹 Clear", use_container_width=True):
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

            with chat_container:
                with st.chat_message("user"):
                    st.write(clean_question)

                with st.chat_message("assistant"):
                    with st.spinner(
                        "Retrieving authorised knowledge chunks and generating answer..."
                    ):
                        try:
                            start_time = time.perf_counter()
                            result = ask_backend(
                                clean_question,
                                department_filter,
                                file_type_filter,
                            )
                        except requests.exceptions.HTTPError as error:
                            latency_seconds = time.perf_counter() - start_time
                            write_query_log(
                                question=clean_question,
                                department_filter=department_filter,
                                file_type_filter=file_type_filter,
                                status="api_error",
                                sources=[],
                                latency_seconds=latency_seconds,
                            )
                            assistant_message = {
                                "role": "assistant",
                                "content": f"API returned an error: {error.response.text}",
                                "sources": [],
                                "context": "",
                                "status": "api_error",
                            }
                        except requests.exceptions.RequestException as error:
                            latency_seconds = time.perf_counter() - start_time
                            write_query_log(
                                question=clean_question,
                                department_filter=department_filter,
                                file_type_filter=file_type_filter,
                                status="connection_error",
                                sources=[],
                                latency_seconds=latency_seconds,
                            )
                            assistant_message = {
                                "role": "assistant",
                                "content": f"Could not connect to the FastAPI backend: {error}",
                                "sources": [],
                                "context": "",
                                "status": "connection_error",
                            }
                        else:
                            latency_seconds = time.perf_counter() - start_time
                            answer_status = classify_answer_status(
                                result["answer"],
                                result["sources"]
                            )
                            write_query_log(
                                question=clean_question,
                                department_filter=department_filter,
                                file_type_filter=file_type_filter,
                                status=answer_status,
                                sources=result["sources"],
                                latency_seconds=latency_seconds
                            )
                            context_text = (
                                f"Access context: {result['role']} / {result['department']} | "
                                f"Search department: {department_filter or 'ACL-permitted shared scope'} | "
                                f"File type: {file_type_filter}"
                            )

                            assistant_message = {
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"],
                                "context": context_text,
                                "status": answer_status,
                            }

                    show_status_message(assistant_message["status"])
                    st.write(assistant_message["content"])

                    if assistant_message.get("sources") or assistant_message.get("context"):
                        meta_col1, meta_col2 = st.columns(2)
                        if assistant_message.get("sources"):
                            with meta_col1:
                                with st.expander("📑 View Sources"):
                                    for source in assistant_message["sources"]:
                                        st.code(source, language=None)
                        if assistant_message.get("context"):
                            with meta_col2:
                                with st.expander("🔍 Query Context"):
                                    st.caption(assistant_message["context"])

            st.session_state["chat_messages"].append(assistant_message)
            st.session_state["chat_is_processing"] = False
            st.rerun()


elif selected_page == "Settings":
    st.header("System Settings")
    st.caption("Admin-only retrieval and guardrail configuration placeholder.")

    st.info(
        "Later this page will include model selection, Top-K settings, "
        "ACL enforcement mode, visual extraction settings, and prompt guardrails."
    )