# REQ_F005: Streamlit Admin Web Portal for manager/admin workflows
# REQ_F004: Displays cited answers returned by the shared FastAPI RAG backend

import json
from pathlib import Path
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/query"
METADATA_PATH = Path("data/simulated/document_metadata.json")

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


def ask_backend(question: str) -> dict:
    """Send one user question and session context to the shared FastAPI RAG backend."""
    response = requests.post(
        API_URL,
        json={
            "question": question,
            "role": st.session_state["role"],
            "department": st.session_state["department"],
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
    st.caption("Dashboard placeholder for retrieval evaluation and system performance.")

    st.info(
        "Later this page will show Top-K Accuracy, miss rate, indexed documents, "
        "latency, and the retrieval miss / improvement log."
    )

elif selected_page in ["KB Management", "KB Status"]:
    st.title(selected_page)

    if st.session_state["role"] == "System Admin":
        st.caption("Global knowledge base management for all departments.")
        st.info(
            "Later this page will show all document metadata, source connectors, "
            "upload/indexing controls, ACL metadata, and visual extraction status."
        )
    elif st.session_state["role"] == "Project Manager":
        st.caption("Department-scoped knowledge base management.")
        st.info(
            "Later this page will show manageable documents for the current department only, "
            "with department-level upload/request controls."
        )
    else:
        st.caption("Department-scoped knowledge base status.")
        st.info(
            "Later this page will show view-only document status for the current user's "
            "allowed department and role."
        )

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
    st.caption("Ask a question through the shared FastAPI RAG backend.")

    question = st.text_input(
        "Ask a question",
        placeholder="Example: What are the password policy requirements?",
    )

    if st.button("Search Knowledge Base"):
        if not question.strip():
            st.warning("Please enter a question before searching.")
        else:
            with st.spinner("Retrieving authorised knowledge chunks and generating answer..."):
                try:
                    result = ask_backend(question.strip())
                except requests.exceptions.HTTPError as error:
                    st.error(f"API returned an error: {error.response.text}")
                except requests.exceptions.RequestException as error:
                    st.error(f"Could not connect to the FastAPI backend: {error}")
                else:
                    st.subheader("Answer")
                    st.write(result["answer"])
                    st.caption(
                        f"Query context: {result['role']} / {result['department']} "
                        "(ACL filtering will be enforced in a later backend slice.)"
                    )

                    st.subheader("Sources")
                    for source in result["sources"]:
                        st.code(source)

elif selected_page == "Settings":
    st.title("Settings")
    st.caption("Admin-only retrieval and guardrail configuration placeholder.")

    st.info(
        "Later this page will include model selection, Top-K settings, "
        "ACL enforcement mode, visual extraction settings, and prompt guardrails."
    )