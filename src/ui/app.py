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
import sys


ROLE_OPTIONS = [
    "System Admin",
    "Project Manager",
    "General Employee",
]

DEPARTMENT_OPTIONS = [
    "IT",
    "Engineering",
    "HR",
    "Security",
    "Operations",
]

ROLE_AWARE_CHAT_PROMPTS = {
    "System Admin": {
        "Password Policy": "What are the password policy requirements?",
        "Security Incident": "What is the security incident reporting procedure?",
        "Annual Leave": "What is the annual leave approval process?",
        "VPN Setup": "How do I set up the company VPN?",
    },
    "Project Manager": {
        "Development Workflow": "What is the software development workflow?",
        "Coding Standards": "What are the Python coding standards?",
        "VPN Setup": "How do I set up the company VPN?",
        "Security Incident": "What is the security incident reporting procedure?",
    },
    "General Employee": {
        "Annual Leave": "What is the annual leave approval process?",
        "Expense Claims": "How do I submit an expense claim?",
        "Security Incident": "What is the security incident reporting procedure?",
        "Onboarding": "What is the employee onboarding process?",
    },
}

DEMO_ACCOUNTS = {
    "admin_jc": {
        "password": "password123",
        "role": "System Admin",
        "department": "IT",
    },
    "pm_engineering": {
        "password": "password123",
        "role": "Project Manager",
        "department": "Engineering",
    },
    "employee_hr": {
        "password": "password123",
        "role": "General Employee",
        "department": "HR",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

UPLOAD_VALIDATE_URL = "http://127.0.0.1:8000/admin/validate-upload"
REINDEX_URL = "http://127.0.0.1:8000/admin/reindex"
API_URL = "http://127.0.0.1:8000/query"
API_HEALTH_URL = "http://127.0.0.1:8000/health"
METADATA_PATH = PROJECT_ROOT / "data/simulated/document_metadata.json"
QUERY_LOG_DB_PATH = PROJECT_ROOT / "data/logs/query_logs.db"
EVALUATION_RESULTS_PATH = PROJECT_ROOT / "data/evaluation/retrieval_eval_results.json"


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
    if st.session_state["role"] in ["System Admin", "Project Manager"]:
        return "KB Management"

    return "KB Status"


def can_access_settings() -> bool:
    """Check whether the current user can access admin-only settings."""
    return st.session_state["role"] == "System Admin"


def load_document_metadata() -> list[dict]:
    """Load simulated document metadata for the KB Management/Status page."""
    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def load_retrieval_evaluation_results() -> dict | None:
    """Load the latest local retrieval evaluation output if it exists."""
    if not EVALUATION_RESULTS_PATH.exists():
        return None

    with EVALUATION_RESULTS_PATH.open("r", encoding="utf-8") as results_file:
        return json.load(results_file)


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

    if role == "System Admin":
        return True

    return (
        role in document["allowed_roles"]
        and department in document["allowed_departments"]
    )


def generate_document_id(documents: list[dict]) -> str:
    """Generate the next local upload document ID."""
    upload_count = sum(
        1 for document in documents
        if document["document_id"].startswith("DOC-UPLOAD-")
    )

    return f"DOC-UPLOAD-{upload_count + 1:03d}"


def normalise_uploaded_filename(filename: str) -> str:
    """Return the local filename used for uploaded simulated documents."""
    return filename.replace(" ", "_")


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


def append_document_metadata(new_document: dict) -> None:
    """Append a new document metadata record to the local metadata JSON file."""
    documents = load_document_metadata()
    documents.append(new_document)

    with METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(documents, metadata_file, indent=2)


def infer_title_from_uploaded_file(uploaded_file) -> str:
    """Infer a default document title from the uploaded filename."""
    return Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ").title()


def metadata_exists_for_filename(filename: str) -> bool:
    """Check whether metadata already exists for an uploaded filename."""
    documents = load_document_metadata()

    return any(
        document["filename"] == filename
        for document in documents
    )


st.set_page_config(
    page_title="Searchable RAG Copilot",
    page_icon="R",
    layout="wide",
)


if not is_logged_in():
    st.title("Searchable RAG Copilot")
    st.caption("Demo sign-in for the standalone Admin Web Portal.")

    username = st.text_input("Username", value="admin_jc")
    password = st.text_input("Password", value="password123", type="password")

    if st.button("Sign In"):
        if login_user(username.strip(), password):
            st.rerun()
        else:
            st.error("Invalid demo username or password.")

    st.stop()


api_online = is_api_online()
api_status_label = "API Online" if api_online else "API Offline"
api_status_color = "#166534" if api_online else "#991b1b"
api_status_background = "#dcfce7" if api_online else "#fee2e2"

st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0 0.65rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    ">
        <div>
            <div style="font-size: 1.1rem; font-weight: 700;">
                Searchable RAG Copilot
            </div>
            <div style="font-size: 0.78rem; color: #64748b;">
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
        ">
            {escape(api_status_label)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.2rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        width: 100%;
        padding: 0.58rem 0.7rem;
        border-radius: 0.3rem;
        cursor: pointer;
        transition: background-color 0.15s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: #eef2f7;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background-color: #e2e8f0;
        font-weight: 600;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label
    [data-testid="stMarkdownContainer"] {
        width: 100%;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {
        display: none;
    }

    [data-testid="stForm"] [data-testid="InputInstructions"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Account")
st.sidebar.caption(escape(st.session_state["user"]))
st.sidebar.markdown(
    f"""
    <div style="
        padding: 0.55rem 0.65rem;
        border: 1px solid #e2e8f0;
        border-radius: 0.35rem;
        background: #ffffff;
        font-size: 0.78rem;
        line-height: 1.65;
        margin-bottom: 0.8rem;
    ">
        <strong>Role</strong><br>
        {escape(st.session_state["role"])}<br>
        <strong>Department</strong><br>
        {escape(st.session_state["department"])}
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

if st.session_state["role"] in ["System Admin", "Project Manager"]:
    page_options.insert(0, "Performance")

if can_access_settings():
    page_options.append("Settings")

selected_page = st.sidebar.radio("Navigation", page_options)

st.sidebar.divider()

if st.sidebar.button("Logout", use_container_width=True):
    logout_user()
    st.rerun()


if selected_page == "Performance":
    st.title("Performance")
    st.caption(
        "Performance dashboard using local query logs, labelled retrieval evaluation, "
        "and clearly marked simulated benchmark placeholders."
    )

    documents = load_document_metadata()
    indexed_document_count = len(documents)
    query_log_summary = read_query_log_summary()
    evaluation_results = load_retrieval_evaluation_results()

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

    st.subheader("Performance Metrics")

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

    st.divider()

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
        st.write(
            "Top-K Accuracy checks whether the expected source document appears "
            "within the top 5 retrieved chunks."
        )
        st.caption(
            "Current Top-K Accuracy and Miss Rate use the latest local labelled "
            "retrieval evaluation result. Prototype benchmark values should be treated "
            "as presentation examples only."
        )

    st.subheader("Retrieval Miss Review Log")

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

    st.subheader("Recent Logged Queries")

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
    st.title(selected_page)

    if st.session_state["role"] == "System Admin":
        st.caption("Global knowledge base management for all departments.")
    elif st.session_state["role"] == "Project Manager":
        st.caption("Department-scoped knowledge base management.")
    else:
        st.caption("Department-scoped knowledge base status.")

    documents = load_document_metadata()
    visible_documents = [
        document for document in documents
        if can_view_document(document)
    ]

    st.divider()

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
            st.caption("Enterprise source integration direction")
            st.markdown("- SharePoint\n- OneNote\n- Manual Upload\n- Batch ZIP")

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
            st.caption("Access is decided before retrieval")
            st.markdown(
                f"- Role: {escape(st.session_state['role'])}\n"
                f"- Department: {escape(st.session_state['department'])}\n"
                f"- Visible documents: {len(visible_documents)}"
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
            st.caption("Document extraction capability")
            st.markdown(
                "- Text extraction: Active for local files\n"
                "- OCR captions: Roadmap\n"
                "- Diagram extraction: Roadmap"
            )
    
    if st.session_state["role"] != "General Employee":

        st.subheader("Document Ingestion")

        if "upload_message" not in st.session_state:
            st.session_state["upload_message"] = ""

        with st.container(border=True):
            if st.session_state["role"] in ["System Admin", "Project Manager"]:
                st.markdown("**Real Local TXT/PDF/DOCX Upload**")
                st.caption(
                    "Uploads a TXT file into data/simulated and appends trusted metadata. "
                    "Rebuild the vector index after upload before searching the new document."
                )

                if "upload_form_version" not in st.session_state:
                    st.session_state["upload_form_version"] = 0

                upload_form_version = st.session_state["upload_form_version"]

                uploaded_file = st.file_uploader(
                    "TXT, PDF, or DOCX file",
                    type=["txt", "pdf", "docx"],
                    key=f"upload_file{upload_form_version}",
                )

                title_key = prepare_upload_title_state(uploaded_file, upload_form_version)

                with st.form(f"real_txt_upload_form_{upload_form_version}"):

                    title = st.text_input(
                        "Document title",
                        key=title_key,
                        help="Auto-filled from the uploaded filename. Admin may edit it.",
                    )
                    if st.session_state["role"] == "System Admin":
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
                    tags_text = st.text_input(
                        "Tags",
                        value="policy, internal",
                        help="Separate tags with commas.",
                        key=f"txt_upload_tags_{upload_form_version}",
                    )
                    if st.session_state["role"] == "System Admin":
                        allowed_roles = st.multiselect(
                            "Allowed roles",
                            ROLE_OPTIONS,
                            default=["System Admin"],
                            key=f"txt_upload_roles_{upload_form_version}",
                        )
                    else:
                        allowed_roles = st.multiselect(
                            "Allowed roles",
                            ["Project Manager", "General Employee"],
                            default=["Project Manager"],
                            key=f"txt_upload_roles_{upload_form_version}",
                        )
                    if st.session_state["role"] == "System Admin":
                        allowed_departments = st.multiselect(
                            "Allowed departments",
                            DEPARTMENT_OPTIONS,
                            default=[department],
                            key=f"txt_upload_departments_{upload_form_version}",
                        )
                    else:
                        allowed_departments = st.multiselect(
                            "Allowed departments",
                            [st.session_state["department"]],
                            default=[st.session_state["department"]],
                            key=f"txt_upload_departments_{upload_form_version}",
                        )

                    submitted_upload = st.form_submit_button("Save File + Metadata")

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
                                "chunk_id": "pending",
                                "visual_extraction_status": get_visual_extraction_status(file_type),
                            }

                            append_document_metadata(new_document)

                            st.session_state["upload_message"] = (
                                f"Saved {filename} and appended metadata record "
                                f"{new_document['document_id']}. Rebuild ChromaDB before searching it."
                            )

                            st.session_state["upload_form_version"] += 1
                            st.rerun()
                if st.session_state["role"] == "System Admin":
                    st.markdown("**Local Vector Index**")
                    st.caption(
                        "Rebuild after adding or changing simulated documents so ChromaDB "
                        "can retrieve the latest content."
                    )

                
                    if st.button("Rebuild Local Vector Index"):
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

    st.subheader("Document Index & Permission Metadata")

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

        filter_columns = st.columns(3)

        with filter_columns[0]:
            selected_department = st.selectbox(
                "Visible Document Department",
                ["All"] + department_options,
            )

        with filter_columns[1]:
            selected_category = st.selectbox(
                "Category",
                ["All"] + category_options,
            )

        with filter_columns[2]:
            selected_source = st.selectbox(
                "Source",
                ["All"] + source_options,
            )

        filtered_documents = visible_documents

        if selected_department != "All":
            filtered_documents = [
                document for document in filtered_documents
                if document["department"] == selected_department
            ]

        if selected_category != "All":
            filtered_documents = [
                document for document in filtered_documents
                if document["category"] == selected_category
            ]

        if selected_source != "All":
            filtered_documents = [
                document for document in filtered_documents
                if document["source"] == selected_source
            ]

        table_rows = [
            {
                "Document": document["title"],
                "Source": document["source"],
                "Department": document["department"],
                "Category": document["category"],
                "Allowed Access": ", ".join(document["allowed_roles"]),
                "Visuals": document["visual_extraction_status"],
            }
            for document in filtered_documents
        ]

        st.dataframe(
            table_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Selected Document Details")

        if not filtered_documents:
            st.info("No documents match the selected filters.")
        else:
            selected_title = st.selectbox(
                "Document",
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
                    st.write(f"Document ID: {selected_document['document_id']}")
                    st.write(f"Filename: {selected_document['filename']}")
                    st.write(f"File type: {selected_document['file_type']}")
                    st.write(f"Uploaded by: {selected_document['uploaded_by']}")
                    st.write(f"Uploaded at: {selected_document['uploaded_at']}")

            with detail_columns[1]:
                with st.container(border=True):
                    st.markdown("**Access & Extraction**")
                    st.write(f"Tags: {', '.join(selected_document['tags'])}")
                    st.write(
                        "Allowed roles: "
                        f"{', '.join(selected_document['allowed_roles'])}"
                    )
                    st.write(
                        "Allowed departments: "
                        f"{', '.join(selected_document['allowed_departments'])}"
                    )
                    st.write(f"Chunk ID: {selected_document['chunk_id']}")
                    st.write(
                        "Visual extraction status: "
                        f"{selected_document['visual_extraction_status']}"
                    )


elif selected_page == "Chat":
    st.title("Chat")
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

    st.caption("Ask questions through the shared FastAPI RAG backend.")

    with st.container(border=True):
        filter_columns = st.columns([1, 1, 2])
        if st.session_state["role"] == "System Admin":
            with filter_columns[0]:
                department_filter = st.selectbox(
                    "Department",
                    ["All"] + available_departments,
                )

            with filter_columns[1]:
                file_type_filter = st.selectbox(
                    "File Type",
                    ["All"] + available_file_types,
                )

            filter_status = (
                f"Selected filter: Department = {department_filter}, "
                f"File Type = {file_type_filter}"
            )

        elif st.session_state["role"] == "Project Manager":
            department_filter = None

            with filter_columns[0]:
                st.text_input(
                    "Department",
                    value=(
                        f"{st.session_state['department']} + "
                        "ACL-permitted shared documents"
                    ),
                    disabled=True,
                )

            with filter_columns[1]:
                file_type_filter = st.selectbox(
                    "File Type",
                    ["All"] + available_file_types,
                )

            filter_status = (
                f"Selected filter: Department = {st.session_state['department']} + "
                f"ACL-permitted shared documents, File Type = {file_type_filter}"
            )

        else:
            department_filter = None
            file_type_filter = "All"

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

    st.divider()

    if st.button("Clear Chat"):
        st.session_state["chat_messages"] = []
        st.rerun()

    chat_container = st.container(height=420, border=True)

    with chat_container:
        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and message.get("status"):
                    show_status_message(message["status"])

                st.write(message["content"])

                if message.get("sources"):
                    st.caption("Sources")
                    for source in message["sources"]:
                        st.code(source)

                if message.get("context"):
                    st.caption(message["context"])

    chat_is_processing = st.session_state.get("chat_is_processing", False)

    st.caption("Suggested questions")

    example_prompts = ROLE_AWARE_CHAT_PROMPTS[st.session_state["role"]]
    example_columns = st.columns(len(example_prompts))

    for column, (label, prompt) in zip(example_columns, example_prompts.items()):
        with column:
            st.button(
                label,
                key=f"example_prompt_{label}",
                on_click=select_example_chat_prompt,
                args=(prompt,),
                use_container_width=True,
                disabled=chat_is_processing,
            )

    if st.session_state["role"] == "General Employee":
        st.caption("Access control demonstration")
        st.button(
            "Try Restricted IT Policy",
            key="acl_demo_prompt",
            on_click=select_example_chat_prompt,
            args=("What are the password policy requirements?",),
            disabled=chat_is_processing,
        )

    with st.form("chat_question_form"):
        question_columns = st.columns([6, 1])

        with question_columns[0]:
            st.text_input(
                "Ask a question about the knowledge base",
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
            )

    question = st.session_state.pop("pending_chat_question", None)

    if question:
        clean_question = question.strip()

        if not clean_question:
            st.session_state["chat_is_processing"] = False
            st.warning("Please enter a question before searching.")
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

                    if assistant_message.get("sources"):
                        st.caption("Sources")
                        for source in assistant_message["sources"]:
                            st.code(source)

                    if assistant_message.get("context"):
                        st.caption(assistant_message["context"])

            st.session_state["chat_messages"].append(assistant_message)
            st.session_state["chat_is_processing"] = False
            st.rerun()


elif selected_page == "Settings":
    st.title("Settings")
    st.caption("Admin-only retrieval and guardrail configuration placeholder.")

    st.info(
        "Later this page will include model selection, Top-K settings, "
        "ACL enforcement mode, visual extraction settings, and prompt guardrails."
    )
