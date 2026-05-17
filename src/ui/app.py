# REQ_F005: Streamlit Admin Web Portal for manager/admin workflows
# REQ_F004: Displays cited answers returned by the shared FastAPI RAG backend

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/query"


def ask_backend(question: str) -> dict:
    """Send one user question to the shared FastAPI RAG backend."""
    response = requests.post(
        API_URL,
        json={"question": question},
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="Searchable RAG Copilot",
    page_icon="R",
    layout="wide",
)

st.title("Searchable RAG Copilot")
st.caption("Admin Web Portal proof of connection to the shared FastAPI RAG backend.")

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

                st.subheader("Sources")
                for source in result["sources"]:
                    st.code(source)