# Configuration Guide

The application reads secrets and provider defaults from `.env`. Some
administrator settings are also stored in SQLite and take precedence over the
matching `.env` provider defaults.

## Safe Configuration Workflow

```bash
cp .env.example .env
chmod 600 .env
```

Rules:

- never commit `.env`;
- never paste secrets into README files, tickets, screenshots or chat;
- use company-owned resources and credentials for company testing;
- rotate any key that has been exposed outside the approved secret channel;
- restart FastAPI after changing `.env` values;
- inspect **Settings** after restart because SQLite runtime settings can override
  provider selections.

## Local Application Variables

| Variable | Example/default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP server. |
| `OLLAMA_MODEL` | `mistral` | Local answer model. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model used by both vector providers. |
| `EMBEDDING_BACKEND` | `local` | Must remain `local` in this build. |
| `CHROMA_DB_PATH` | `./data/chroma_db` | Local Chroma persistence directory. |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | Chroma collection name. |
| `DOCUMENTS_PATH` | `./data/simulated` | Local ETL working document directory. |
| `API_HOST` | `127.0.0.1` | Documented API bind host; launch command remains authoritative. |
| `API_PORT` | `8000` | Documented API port. |
| `API_BASE_URL` | `http://127.0.0.1:8000` | FastAPI base URL used by Streamlit. |

## Provider Switches

| Variable | Allowed values | Current behavior |
|---|---|---|
| `STORAGE_BACKEND` | `local`, `azure_blob` | Selects file storage for new backend-owned uploads. |
| `VECTOR_BACKEND` | `chroma`, `azure_search` | Selects the active retrieval index. |
| `EMBEDDING_BACKEND` | `local` | Azure embeddings are not implemented. |
| `LLM_BACKEND` | `ollama`, `azure_openai` | Selects answer/rewrite/advisor model. |

The providers do not automatically fail over. Chroma and Azure AI Search can
contain different index states. Switching vector provider requires a deliberate
rebuild or verification that the target index is current.

## Retrieval Variables

| Variable | Default | Purpose |
|---|---:|---|
| `TOP_K` | `5` | Maximum number of chunks accepted for a query before deduplicating citations. |
| `MINIMUM_RELEVANCE_THRESHOLD` | `0.25` | Rejects retrieved chunks below the normalized similarity threshold. |
| `GUARDRAIL_PROMPT` | See `.env.example` | Requires grounded answers from authorized documents. |

Top-K is a chunk limit, not a document limit. Five chunks can come from one
source or several sources. Chunks below the threshold are removed, so the final
answer can cite fewer than five chunks or sources.

## Azure Blob Variables

| Variable | Where to obtain it |
|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Portal → Storage account → Security + networking → Access keys → Connection string. |
| `AZURE_STORAGE_CONTAINER_NAME` | Storage account → Data storage → Containers. Use a private lowercase container such as `kb-documents`. |

The current implementation uses Shared Key connection strings. Treat the entire
connection string as a password. A future production version should prefer
managed identity/RBAC instead of long-lived account keys.

## Azure AI Search Variables

| Variable | Where to obtain it |
|---|---|
| `AZURE_SEARCH_ENDPOINT` | Search service → Overview. Format: `https://<name>.search.windows.net`. |
| `AZURE_SEARCH_ADMIN_KEY` | Search service → Settings → Keys → Primary or secondary admin key. |
| `AZURE_SEARCH_INDEX_NAME` | Application-controlled name, normally `rag-copilot-documents`. |

The backend creates/manages the configured index during rebuild. The admin key
is required by the current implementation because it creates, uploads and
deletes index content. Do not substitute a query-only key.

## Azure OpenAI-Compatible Foundry Variables

| Variable | Where to obtain it |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Foundry resource/deployment endpoint. This project expects the OpenAI v1 base ending in `/openai/v1`. |
| `AZURE_OPENAI_API_KEY` | Foundry resource endpoint/key area. |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | The deployment name selected when deploying the model, for example `rag-gpt-5.4-nano`. |

The deployment name is not always identical to the model family name. The code
passes the deployment name in the OpenAI `model` field.

## Microsoft Graph Variables

| Variable | Example | Purpose |
|---|---|---|
| `GRAPH_CONNECTOR_ENABLED` | `false` | Enables OneDrive and OneNote controls. |
| `GRAPH_CLIENT_ID` | `<GUID>` | Entra app registration Application (client) ID. |
| `GRAPH_AUTHORITY` | `https://login.microsoftonline.com/consumers` | Personal-account authority used by the current setup. |
| `GRAPH_SCOPES` | `User.Read Files.Read Notes.Read` | Delegated read permissions requested by MSAL. |
| `GRAPH_ONEDRIVE_ROOT_PATH` | `/Enterprise Knowledge Base` | Limits scans to one OneDrive folder. |
| `GRAPH_ONENOTE_NOTEBOOK_FILTER` | `Enterprise Knowledge Base` | Limits scans to one notebook name. |
| `GRAPH_ACCESS_TOKEN` | unset | Optional manual debugging fallback only. |

For a company single-tenant setup, use:

```dotenv
GRAPH_AUTHORITY=https://login.microsoftonline.com/<COMPANY_TENANT_ID>
```

There is no Graph API key. `GRAPH_CLIENT_ID` identifies the public client; the
device-code flow obtains delegated access and refresh tokens after user sign-in.

## Complete Local Profile

```dotenv
STORAGE_BACKEND=local
VECTOR_BACKEND=chroma
EMBEDDING_BACKEND=local
LLM_BACKEND=ollama
GRAPH_CONNECTOR_ENABLED=false
```

Required supporting values:

```dotenv
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=rag_documents
DOCUMENTS_PATH=./data/simulated
```

## Complete Hybrid Azure Profile

```dotenv
STORAGE_BACKEND=azure_blob
VECTOR_BACKEND=azure_search
EMBEDDING_BACKEND=local
LLM_BACKEND=azure_openai

AZURE_STORAGE_CONNECTION_STRING=<SECRET>
AZURE_STORAGE_CONTAINER_NAME=kb-documents

AZURE_SEARCH_ENDPOINT=https://<SEARCH_SERVICE>.search.windows.net
AZURE_SEARCH_ADMIN_KEY=<SECRET>
AZURE_SEARCH_INDEX_NAME=rag-copilot-documents

AZURE_OPENAI_ENDPOINT=https://<FOUNDRY_RESOURCE>.services.ai.azure.com/openai/v1
AZURE_OPENAI_API_KEY=<SECRET>
AZURE_OPENAI_CHAT_DEPLOYMENT=rag-gpt-5.4-nano
```

Add the Graph variables only when OneDrive/OneNote ingestion is required.

## SQLite Runtime Overrides

The Streamlit **Settings** page saves active provider and retrieval settings in
the `app_settings` table. The backend reads these before `.env` defaults.

Consequences:

- changing `.env` may not change the visible active provider;
- vector-backend changes can remain pending until a rebuild succeeds;
- the Settings page is the final check of active behavior;
- deleting SQLite to remove overrides is unsafe because the same database holds
  metadata, users, versions, chat memory, jobs and usage records.

Use the application settings workflow instead of manually editing SQLite.

## Configuration Verification

Token-free backend health:

```bash
curl http://127.0.0.1:8000/health
```

Graph identity check after device login:

```bash
python - <<'PY'
from src.connectors.graph_client import get_current_graph_user
print(get_current_graph_user()["displayName"])
PY
```

Configured LLM class check. This does not call the model:

```bash
python - <<'PY'
from src.rag.llm_factory import create_chat_llm
print(type(create_chat_llm()).__name__)
PY
```

Use the Settings **Test** button only when one real model request and its token
cost are acceptable.
