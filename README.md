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

## Start Here: Choose a Setup Path

You do not need Azure, Microsoft Graph or Teams to run the core system. Choose
one path before installing:

### Path A: Fully Local

Choose this when Azure resources are unavailable or when evaluating the system
without cloud cost.

1. Complete [Installation](docs/INSTALLATION.md).
2. Copy `.env.example` to `.env` and keep its default local provider values.
3. Install/start Ollama and pull Mistral.
4. Start FastAPI and Streamlit using the commands in this README.
5. Build the initial Chroma index from the portal.

Detailed environment values: [Configuration - Complete Local Profile](docs/CONFIGURATION.md#complete-local-profile).

On a fresh installation, leaving the copied `.env.example` provider values
unchanged selects:

```dotenv
STORAGE_BACKEND=local
VECTOR_BACKEND=chroma
EMBEDDING_BACKEND=local
LLM_BACKEND=ollama
GRAPH_CONNECTOR_ENABLED=false
```

This means documents stay on the local machine, ChromaDB performs retrieval,
Mistral answers through Ollama, and OneDrive/OneNote controls remain disabled.
No Azure account or Microsoft sign-in is required.

### Path B: Hybrid Azure

Choose this to reproduce the current cloud-connected setup while keeping
FastAPI, Streamlit, SQLite and embeddings on the company test machine.

1. Complete [Installation](docs/INSTALLATION.md).
2. Follow [Azure Setup](docs/AZURE_SETUP.md) to create/configure:
   - Azure Blob Storage;
   - Azure AI Search Free;
   - a Foundry `gpt-5.4-nano` deployment.
3. Copy `.env.example` to `.env`, add the company-owned Azure values and select
   the hybrid providers.
4. Start FastAPI and Streamlit.
5. Open **Settings** and verify the active providers before rebuilding the
   Azure Search index.

Detailed environment values: [Configuration - Complete Hybrid Azure Profile](docs/CONFIGURATION.md#complete-hybrid-azure-profile).

Microsoft Graph is optional. The hybrid Azure profile works without OneDrive or
OneNote when `GRAPH_CONNECTOR_ENABLED=false`; documents can still enter through
manual upload and Batch ZIP.

### Optional: OneDrive and OneNote

After either application profile is running, follow
[Graph Connector Setup](docs/GRAPH_CONNECTOR_SETUP.md) to:

1. validate the Microsoft account with Graph Explorer;
2. create the project's Entra app registration;
3. consent `User.Read`, `Files.Read` and `Notes.Read`;
4. configure the OneDrive root and OneNote notebook;
5. run device-code login and verify silent token refresh;
6. scan, stage, review, approve and index connector content.

Graph Explorer consent and the project's Entra app consent are separate. The
guide explains both processes.

### Optional: Teams Client

The Streamlit portal already includes chat. Add the Teams-style client only when
the company wants to test a second frontend. Follow the
[Teams client guide](teams_bot/AgentsToolkitProjects/teams-chat-bot/README.md).
The Teams client needs only the FastAPI URL; Azure and Graph credentials remain
in the backend `.env`.

## Guide Map

| Goal | Read this |
|---|---|
| Install WSL, Python, OCR, Ollama and start the application | [Installation](docs/INSTALLATION.md) |
| Configure providers, prompts and controlled-UAT accounts | [Configuration](docs/CONFIGURATION.md) |
| Obtain Blob, Search and Foundry endpoints/keys | [Azure Setup](docs/AZURE_SETUP.md) |
| Configure Graph Explorer, Entra, OneDrive and OneNote | [Graph Connector Setup](docs/GRAPH_CONNECTOR_SETUP.md) |
| Learn every role-visible portal workflow | [User Manual](docs/USER_MANUAL.md) |
| Operate upload, review, indexing, versions and connectors | [Administrator Operations](docs/ADMIN_OPERATIONS.md) |
| Add labelled queries and measure accuracy/miss rate | [Evaluation Guide](docs/EVALUATION_GUIDE.md) |
| Run a controlled company test | [Company UAT Guide](docs/COMPANY_UAT_GUIDE.md) |
| Verify the fixed handover release and package contents | [Handover Manifest](HANDOVER_MANIFEST.md) |
| Diagnose errors | [Troubleshooting](docs/TROUBLESHOOTING.md) |
| Review restrictions before sharing access | [Security and Limitations](docs/SECURITY_AND_LIMITATIONS.md) |
| Back up or restore system state | [Backup and Recovery](docs/BACKUP_AND_RECOVERY.md) |
| Understand the complete data flow | [Architecture](docs/ARCHITECTURE.md) |

## Supported Runtime Profiles

| Component | Local profile | Hybrid Azure profile |
|---|---|---|
| Document storage | Local filesystem | Azure Blob Storage plus local ETL copy |
| Vector index | ChromaDB | Azure AI Search Free |
| Embeddings | Local `all-MiniLM-L6-v2` | Local `all-MiniLM-L6-v2` |
| Answer model | Ollama/Mistral | Azure OpenAI `gpt-5.4-nano` deployment |
| Metadata and governance | SQLite | SQLite |
| OneDrive/OneNote | Disabled or optional | Microsoft Graph delegated read access |
| SharePoint | Not implemented | Future enterprise integration |

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
sudo apt install -y python3 python3-venv python3-pip git curl tesseract-ocr
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
git clone https://github.com/Jameschoong7/searchable-rag-copilot-system.git
cd searchable-rag-copilot-system

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Initial model downloads can take time. The dependency file selects CPU-only
PyTorch so a normal company test installation does not download the CUDA/NVIDIA
toolchain. The local embedding model is used by both ChromaDB and Azure AI
Search.

## 4. Configure the Environment

```bash
cp .env.example .env
```

Open `.env` in an editor and choose one of these paths:

- **Local:** retain `STORAGE_BACKEND=local`, `VECTOR_BACKEND=chroma` and
  `LLM_BACKEND=ollama`. Continue with
  [Configuration - Complete Local Profile](docs/CONFIGURATION.md#complete-local-profile).
- **Hybrid Azure:** set `STORAGE_BACKEND=azure_blob`,
  `VECTOR_BACKEND=azure_search`, and `LLM_BACKEND=azure_openai`, then provide
  company-owned Azure values obtained through
  [Azure Setup](docs/AZURE_SETUP.md). Continue with
  [Configuration - Complete Hybrid Azure Profile](docs/CONFIGURATION.md#complete-hybrid-azure-profile).

OneDrive and OneNote are not required for either initial path. Enable them later
through [Graph Connector Setup](docs/GRAPH_CONNECTOR_SETUP.md).

Never commit `.env`, Azure keys, connection strings or Graph token caches.
See [Configuration](docs/CONFIGURATION.md) for every variable and
[Azure Setup](docs/AZURE_SETUP.md) for where each Azure value comes from.

Important: settings saved through the portal are stored in SQLite and override
the equivalent provider defaults from `.env`. Verify the **Settings** page when
the runtime profile differs from the file.

## 5. Prepare the Local LLM Profile

Install Ollama using its supported installer. First check whether its background
service is already running:

```bash
curl http://127.0.0.1:11434/api/tags
```

If that request fails, start Ollama in its own terminal:

```bash
ollama serve
```

Then, in another terminal, download the configured Mistral 7B model:

```bash
ollama pull mistral
```

Do not start a second `ollama serve` process when the installer already started
the service. Ollama is not required when `LLM_BACKEND=azure_openai` is active,
but keeping it available provides an intentional local fallback.

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
4. On a fresh clone, select **Confirm full rebuild** and click **Full Rebuild**.
5. Wait for the job to report success before asking the first question.
6. After the initial index exists, use **Run Update for Pending Documents** for
   routine uploads and refreshed versions.
7. Run another full rebuild only for a deliberate vector-backend change or
   recovery operation.

The repository seeds document metadata, but it does not distribute a generated
Chroma database. Therefore, the first full rebuild is required even when no
documents are shown as pending.

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

The primary workflow is:

1. Install [Visual Studio Code](https://code.visualstudio.com/).
2. Install [Node.js 20 or 22](https://nodejs.org/en/download) in the environment
   where the bot will run, then verify `node --version`.
3. Install the [Microsoft 365 Agents Toolkit](https://learn.microsoft.com/microsoftteams/platform/toolkit/install-agents-toolkit)
   extension (`TeamsDevApp.ms-teams-vscode-extension`). If VS Code asks, install
   the extension in the active WSL window as well.
4. From the repository root, open the bot folder as its own VS Code workspace:

   ```bash
   code teams_bot/AgentsToolkitProjects/teams-chat-bot
   ```

   Opening only the repository root with `code .` does not expose the nested
   bot folder's `.vscode/launch.json` Run configuration.
5. Set `RAG_API_BASE_URL=http://127.0.0.1:8000` in `env/.env.playground`.
6. Start FastAPI from the repository root.
7. Open **Run and Debug**, select **Debug in Microsoft 365 Agents Playground**,
   and press **Run** (`F5`).

Agents Toolkit checks prerequisites, prepares the local Playground environment,
starts the bot, opens the Playground and attaches the debugger. A Microsoft 365
tenant is not required for this local simulation. See the
[Teams client guide](teams_bot/AgentsToolkitProjects/teams-chat-bot/README.md)
for the command-line alternative. Port `9239` is Node DevTools, not the chatbot
interface.

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

- [Company Handover Manifest](HANDOVER_MANIFEST.md)
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
