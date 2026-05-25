# REQ_F005: Streamlit Admin Web Portal for manager/admin workflows
# REQ_F004: Displays cited answers returned by the shared FastAPI RAG backend

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/query"

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