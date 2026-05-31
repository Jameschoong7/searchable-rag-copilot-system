# REQ_F005: Streamlit Admin Web Portal for manager/admin workflows
# REQ_F004: Displays cited answers returned by the shared FastAPI RAG backend

from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/query"
METADATA_PATH = Path("data/simulated/document_metadata.json")
QUERY_LOG_DB_PATH = Path("data/logs/query_logs.db")

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
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


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

        return {
            "total_queries": summary_row[0],
            "average_latency": summary_row[1],
            "permission_blocks": summary_row[2] or 0,
            "unresolved_queries": summary_row[3] or 0,
            "recent_queries": recent_rows,
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

st.title("Searchable RAG Copilot")
st.sidebar.subheader("Current Session")
st.sidebar.write(f"User: {st.session_state['user']}")
st.sidebar.write(f"Role: {st.session_state['role']}")
st.sidebar.write(f"Department: {st.session_state['department']}")

kb_page_label = get_kb_page_label()

page_options = [
    "Performance",
    kb_page_label,
    "Chat",
]

if can_access_settings():
    page_options.append("Settings")

selected_page = st.sidebar.radio("Navigation", page_options)

if st.sidebar.button("Logout"):
    logout_user()
    st.rerun()


if selected_page == "Performance":
    st.title("Performance")
    st.caption(
        "Simulated retrieval evaluation dashboard. Later this will be backed by "
        "labelled query tests and real request logs."
    )

    documents = load_document_metadata()
    indexed_document_count = len(documents)
    query_log_summary = read_query_log_summary()

    st.subheader("Planned Benchmark Metrics")

    metric_columns = st.columns(4)

    with metric_columns[0]:
        st.metric(
            "Target Time-to-First-Answer",
            "1.8s",
            "Target: < 10s",
        )

    with metric_columns[1]:
        st.metric(
              "Simulated Top-K Accuracy (K=5)",
              "91.2%",
              "95 / 104 test queries",
        )

    with metric_columns[2]:
        st.metric(
            "Simulated Miss Rate",
            "8.8%",
            "9 queries need review",
        )

    with metric_columns[3]:
        st.metric(
            "Indexed Documents",
            f"{indexed_document_count}",
            "Simulated KB records",
        )

    st.caption(
        "Benchmark cards above are simulated presentation placeholders until a "
        "labelled evaluation dataset exists."
    )

    st.subheader("Live Query Signals")

    live_metric_columns = st.columns(4)

    with live_metric_columns[0]:
        st.metric(
            "Logged Queries",
            query_log_summary["total_queries"]
        )

    with live_metric_columns[1]:
        st.metric(
            "Average Local Latency",
            f"{query_log_summary['average_latency']:.2f}s",
        )

    with live_metric_columns[2]:
        st.metric(
            "Permission Blocks",
            query_log_summary["permission_blocks"],
        )

    with live_metric_columns[3]:
        st.metric(
            "Not Found / Errors",
            query_log_summary["unresolved_queries"],
        )

    st.divider()

    chart_columns = st.columns([2, 1])

    with chart_columns[0]:
        st.subheader("Query Response Latency - Past 7 Days")
        latency_data = {
            "Average latency (seconds)": [1.1, 1.4, 0.9, 1.7, 1.3, 1.5, 1.8]
        }
        st.line_chart(latency_data)

    with chart_columns[1]:
        st.subheader("Benchmark Definition")
        st.write(
            "Top-K Accuracy means the correct source document appears within "
            "the top 5 retrieved chunks."
        )
        st.write("Current simulated benchmark: 95 correct / 104 labelled queries.")
        st.write(
            "The dashboard is simulated until a labelled evaluation set and "
            "request logging are implemented."
        )

    st.subheader("Retrieval Miss / Improvement Log")

    miss_rows = [
        {
            "Query ID": "Q-018",
            "User Query": "VPN profile missing",
            "Issue": "Correct chunk ranked #7",
            "Next Enhancement": "Improve metadata tags or increase K",
        },
        {
            "Query ID": "Q-041",
            "User Query": "Access approval flow",
            "Issue": "Diagram text missing",
            "Next Enhancement": "OCR + diagram caption extraction",
        },
        {
            "Query ID": "Q-073",
            "User Query": "HR claim limit",
            "Issue": "Outdated source PDF",
            "Next Enhancement": "Re-index updated SharePoint file",
        },
    ]

    st.dataframe(
        miss_rows,
        use_container_width=True,
        hide_index=True,
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
            st.caption("Production direction: documents sync from enterprise sources.")
            st.write("SharePoint Auto Sync: Connected / simulated")
            st.write("OneNote Sync: Connected / simulated")
            st.write("Manual Upload: Available")
            st.write("Batch ZIP: Planned")

    with summary_columns[1]:
        with st.container(border=True):
            st.markdown("**Document Metadata & ACL**")
            st.caption("Access is decided before retrieval.")
            st.write(f"Current role: {st.session_state['role']}")
            st.write(f"Current department: {st.session_state['department']}")
            st.write(f"Visible documents: {len(visible_documents)}")

    with summary_columns[2]:
        with st.container(border=True):
            st.markdown("**Visual Content Handling**")
            st.caption("Supports the A3 diagram/OCR requirement.")
            st.write("Text extraction: Available")
            st.write("OCR captions: Planned")
            st.write("Diagram extraction: Planned")

    st.subheader("Document Ingestion")

    if "upload_message" not in st.session_state:
        st.session_state["upload_message"] = ""

    with st.container(border=True):
        if st.session_state["role"] == "General Employee":
            st.info(
                "Knowledge base upload controls are hidden because General Employee "
                "is a view-only role."
            )
        else:
            if st.session_state["role"] == "System Admin":
                st.caption(
                    "System Admin can simulate global document ingestion across departments."
                )
            else:
                st.caption(
                    "Project Manager can simulate department-scoped ingestion for "
                    f"{st.session_state['department']} only."
                )

            upload_columns = st.columns(5)

            with upload_columns[0]:
                if st.button("Upload PDF"):
                    if st.session_state["role"] == "System Admin":
                        st.session_state["upload_message"] = (
                            "Document successfully chunked, embedded, permission-tagged, "
                            "and indexed with global admin permissions."
                        )
                    else:
                        st.session_state["upload_message"] = (
                            "Department-scoped document upload queued for "
                            f"{st.session_state['department']} review and indexing."
                        )

            with upload_columns[1]:
                if st.button("Batch ZIP Upload"):
                    st.session_state["upload_message"] = (
                        "Batch ingestion simulation completed. Multiple files were extracted, "
                        "metadata was inherited, and indexing was queued."
                    )

            with upload_columns[2]:
                if st.button("PDF with Diagram/OCR"):
                    st.session_state["upload_message"] = (
                        "Document indexed with OCR and diagram-caption extraction enabled. "
                        "Embedded flowcharts are now searchable in the simulation."
                    )

            with upload_columns[3]:
                if st.button("File Too Large"):
                    st.session_state["upload_message"] = (
                        "Upload rejected. Detected file size exceeds the simulated 50 MB limit."
                    )

            with upload_columns[4]:
                if st.button("Network Error"):
                    st.session_state["upload_message"] = (
                        "Ingestion failed. Simulated connection loss during embedding/indexing."
                    )

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
                "Department",
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
            department_filter = st.session_state["department"]

            with filter_columns[0]:
                st.text_input(
                    "Department",
                    value=department_filter,
                    disabled=True,
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

    st.caption(
        "Try: password policy requirements | security incident reporting procedure | "
        "annual leave approval process | VPN setup"
    )

    question = st.chat_input("Ask a question about the knowledge base...")

    if question:
        clean_question = question.strip()

        if not clean_question:
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
            st.rerun()


elif selected_page == "Settings":
    st.title("Settings")
    st.caption("Admin-only retrieval and guardrail configuration placeholder.")

    st.info(
        "Later this page will include model selection, Top-K settings, "
        "ACL enforcement mode, visual extraction settings, and prompt guardrails."
    )