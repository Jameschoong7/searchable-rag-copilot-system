# Searchable RAG Copilot

Searchable RAG Copilot is an internal knowledge retrieval system with two
standalone clients connected to one FastAPI backend:

- a Streamlit portal for chat, knowledge-base administration, evaluation and
  system settings;
- a Microsoft Teams-style client built with Microsoft 365 Agents Toolkit;
- a shared RAG pipeline that applies document ACL and department filtering
  before retrieved content is sent to the configured language model.

The system supports a fully local profile and a cost-conscious hybrid Azure
profile. SQLite remains the governance source of truth for metadata, ACLs,
versions, jobs, chat history and usage logs in both profiles.

## Supported Runtime Profiles

| Component | Local profile | Hybrid Azure profile |
|---|---|---|
| Document storage | Local filesystem | Azure Blob Storage plus local ETL copy |
| Vector index | ChromaDB | Azure AI Search Free |
| Embeddings | Local `all-MiniLM-L6-v2` | Local `all-MiniLM-L6-v2` |
| Answer model | Ollama/Mistral | Azure OpenAI `gpt-5.4-nano` deployment |
| Metadata and governance | SQLite | SQLite |
| OneDrive/OneNote | Disabled or optional | Microsoft Graph delegated read access |
| SharePoint | Simulated/exported | Simulated/exported |

Provider selection is configuration-driven. It is not automatic failover: if
an active Azure provider fails, the administrator must intentionally switch to
the local provider and ensure that provider's index is current.

## Current Delivery Scope

This repository supports controlled local or company-side user acceptance
testing. The current portal uses seeded demonstration accounts. It does not yet
provide production Entra sign-in for the FastAPI API, public-internet deployment
packaging, a reverse proxy, TLS termination or multi-instance job coordination.
Review [Security and Limitations](docs/SECURITY_AND_LIMITATIONS.md) before
allowing other users to access the system.

## 1. Install Windows Subsystem for Linux

WSL 2 with Ubuntu is the primary supported company setup. Open PowerShell as
Administrator:

```powershell
wsl --install
```

Restart Windows when prompted, open Ubuntu, and create the requested Linux user.
Confirm WSL 2 from PowerShell:

```powershell
wsl --list --verbose
```

Microsoft's current instructions are available in the
[official WSL installation guide](https://learn.microsoft.com/windows/wsl/install).
Native Windows notes are provided in [Installation](docs/INSTALLATION.md).

## 2. Install WSL Prerequisites

Run these commands inside Ubuntu/WSL:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git tesseract-ocr
```

Additional optional prerequisites:

- Ollama and the Mistral model for the local LLM profile;
- Node.js 20 or 22 and Microsoft 365 Agents Toolkit for the Teams client;
- access to company-owned Azure resources for the hybrid profile.

Keep the repository inside the WSL Linux filesystem, such as
`~/projects/searchable-rag-copilot-system`, rather than under `/mnt/c/`, for
better Python and SQLite filesystem behavior.

## 3. Clone and Install the Python Application

```bash
mkdir -p ~/projects
cd ~/projects
git clone <COMPANY_REPOSITORY_URL> searchable-rag-copilot-system
cd searchable-rag-copilot-system

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Initial model downloads can take time. The local embedding model is used by
both ChromaDB and Azure AI Search.

## 4. Configure the Environment

```bash
cp .env.example .env
```

Open `.env` in an editor and choose one of these paths:

- **Local:** retain `STORAGE_BACKEND=local`, `VECTOR_BACKEND=chroma` and
  `LLM_BACKEND=ollama`.
- **Hybrid Azure:** set `STORAGE_BACKEND=azure_blob`,
  `VECTOR_BACKEND=azure_search`, and `LLM_BACKEND=azure_openai`, then provide
  company-owned Azure values.

Never commit `.env`, Azure keys, connection strings or Graph token caches.
See [Configuration](docs/CONFIGURATION.md) for every variable and
[Azure Setup](docs/AZURE_SETUP.md) for where each Azure value comes from.

Important: settings saved through the portal are stored in SQLite and override
the equivalent provider defaults from `.env`. Verify the **Settings** page when
the runtime profile differs from the file.

## 5. Prepare the Local LLM Profile

Install Ollama using its supported installer, then run:

```bash
ollama pull mistral
ollama serve
```

Keep `ollama serve` running in its own terminal. This step is not required when
`LLM_BACKEND=azure_openai` is active, but keeping Ollama available provides an
intentional local fallback.

## 6. Start the Shared Backend

From the repository root in a new WSL terminal:

```bash
source .venv/bin/activate
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Verify it from another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","service":"Searchable RAG Copilot API"}
```

The health endpoint confirms that FastAPI is reachable. It does not make a
billable LLM call and does not prove that the vector index or Graph connector is
ready.

## 7. Start the Streamlit Portal

```bash
source .venv/bin/activate
streamlit run src/ui/app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

Seeded controlled-UAT accounts use the password `password123`:

| Role | Example account |
|---|---|
| System Admin / IT | `admin_jc` |
| Project Manager | `pm_it`, `pm_hr`, `pm_engineering`, `pm_security`, `pm_operations` |
| General Employee | `employee_it`, `employee_hr`, `employee_engineering`, `employee_security`, `employee_operations` |

These are demonstration identities, not production authentication.

## 8. Build the First Search Index

Sign in as `admin_jc`, then:

1. Open **Settings** and verify the active storage, vector and LLM providers.
2. Open **KB Management**.
3. Open **Index Sync**.
4. Use **Update Pending Documents** for routine indexing.
5. Use a full rebuild only for the initial index or a deliberate vector-backend
   change.

An Azure AI Search rebuild deletes and recreates the selected index. Do not run
it casually against a shared environment.

## 9. Optional Microsoft Graph Setup

OneDrive and OneNote use delegated, read-only Microsoft Graph access. Configure
the Entra application and then run:

```bash
source .venv/bin/activate
python scripts/graph_device_login.py
```

The first run displays a device-login URL and code. Later calls use the local
MSAL token cache and attempt silent refresh. See
[Graph Connector Setup](docs/GRAPH_CONNECTOR_SETUP.md), which also includes an
optional Graph Explorer validation and explains why its consent/token is separate
from the project's Entra application.

## 10. Optional Teams Client

```bash
cd teams_bot/AgentsToolkitProjects/teams-chat-bot
npm ci
npm run build
npm run dev:teamsfx:playground
```

Set `RAG_API_BASE_URL=http://127.0.0.1:8000` in
`teams_bot/AgentsToolkitProjects/teams-chat-bot/env/.env.playground` for Agents
Playground, or `env/.env.local` for the local Toolkit environment. The backend
must already be running.

## 11. Verify Retrieval Quality

The evaluation uses the active vector provider and current metadata/index:

```bash
source .venv/bin/activate
python -m src.evaluation.retrieval_eval
```

The labelled test set intentionally can contain misses. Accuracy is a measured
maintenance signal, not a target that should be forced to 100%. See
[Evaluation Guide](docs/EVALUATION_GUIDE.md).

## Documentation

- [Installation](docs/INSTALLATION.md)
- [Configuration](docs/CONFIGURATION.md)
- [Azure Setup](docs/AZURE_SETUP.md)
- [Graph Connector Setup](docs/GRAPH_CONNECTOR_SETUP.md)
- [User Manual](docs/USER_MANUAL.md)
- [Administrator Operations](docs/ADMIN_OPERATIONS.md)
- [Evaluation Guide](docs/EVALUATION_GUIDE.md)
- [Company UAT Guide](docs/COMPANY_UAT_GUIDE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Security and Limitations](docs/SECURITY_AND_LIMITATIONS.md)
- [Backup and Recovery](docs/BACKUP_AND_RECOVERY.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Sample Knowledge Base](sample_knowledge_base/README.md)

## Safe Shutdown

Use `Ctrl+C` in the Streamlit, Uvicorn and Ollama terminals. Wait for active
index, connector or chat jobs to finish first. Submitted job records persist in
SQLite, but running Python background work does not resume automatically after
the backend process stops.
