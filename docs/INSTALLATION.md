# Installation Guide

This guide prepares a clean company test machine. WSL 2 is the primary path;
native Windows is an alternative for teams that cannot use WSL.

## Before You Start

Confirm the tester has:

- Windows 10 version 2004 or later, or Windows 11;
- permission to install WSL, Python packages and optional Microsoft tools;
- at least 16 GB RAM recommended for running Streamlit, FastAPI, local
  embeddings and Ollama together;
- at least 10 GB free disk space for Python packages, local models, indexes and
  working documents;
- Git access to the private repository;
- company-approved Azure credentials if using the hybrid profile;
- no real confidential documents until the controlled-UAT data rules are
  approved.

## Primary Setup: WSL 2

### 1. Install WSL

Open PowerShell as Administrator:

```powershell
wsl --install
```

Restart Windows, open Ubuntu and create the requested Linux username and
password. Confirm the distribution is using WSL 2:

```powershell
wsl --list --verbose
```

If Ubuntu shows version 1:

```powershell
wsl --set-version Ubuntu 2
```

Reference: [Install WSL](https://learn.microsoft.com/windows/wsl/install).

### 2. Install Linux Packages

Inside Ubuntu/WSL:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git curl tesseract-ocr
```

Verify:

```bash
python3 --version
git --version
tesseract --version
```

Python 3.12 is the verified development version. Python 3.10 or newer is the
minimum intended range, but a fresh handover should use Python 3.12 where
available.

### 3. Clone Into the Linux Filesystem

Do not place the repository under `/mnt/c/` unless company policy requires it.
SQLite, Python virtual environments and file-heavy package operations generally
behave better under the WSL Linux filesystem.

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/Jameschoong7/searchable-rag-copilot-system.git
cd searchable-rag-copilot-system
```

### 4. Create the Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify imports without starting cloud services:

```bash
PYTHONDONTWRITEBYTECODE=1 python -c "import src.api.main; print('backend import ok')"
```

### 5. Configure `.env`

```bash
cp .env.example .env
```

Use [Configuration](CONFIGURATION.md) to choose local or hybrid Azure values.
Restrict the file to the current Linux user:

```bash
chmod 600 .env
```

### 6. Optional Local Ollama

Install Ollama using the approved installer for the test machine. Check whether
the installer already started its service:

```bash
curl http://127.0.0.1:11434/api/tags
```

If the request fails, run this in a dedicated terminal:

```bash
ollama serve
```

Then download the configured Mistral 7B model from another terminal:

```bash
ollama pull mistral
```

If Azure OpenAI is active, Ollama is optional. It remains useful for a deliberate
offline fallback. Do not start a second server if Ollama is already listening on
port `11434`.

### 7. Start FastAPI

```bash
cd ~/projects/searchable-rag-copilot-system
source .venv/bin/activate
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Verify from another WSL terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok","service":"Searchable RAG Copilot API"}
```

### 8. Start Streamlit

```bash
cd ~/projects/searchable-rag-copilot-system
source .venv/bin/activate
streamlit run src/ui/app.py
```

Open the displayed URL in the Windows browser. WSL normally exposes
`http://localhost:8501` automatically.

### 9. First Login and Runtime Verification

1. Sign in as `admin_jc` using the controlled-UAT password `password123`.
2. Open **Settings**.
3. Verify the active providers match the intended profile.
4. Expand **LLM Health** details.
5. Use the billable **Test** button only when a real model request is acceptable.
6. Open **KB Management → Index Sync**.
7. On a fresh clone, select **Confirm full rebuild** and click **Full Rebuild**.
8. Wait for the initial rebuild job to succeed. Seed metadata exists on a clean
   installation, but the generated Chroma index does not.
9. Use **Run Update for Pending Documents** for later uploads/refreshes.
10. Ask one known sample question and verify that sources are displayed.

## Native Windows Alternative

Native Windows is supported as an alternative but is less thoroughly exercised
than WSL.

### Prerequisites

Install:

- Git for Windows;
- Python 3.12 with `py` launcher and `pip`;
- Tesseract OCR, then add its installation directory to `PATH`;
- Ollama for Windows if using the local LLM;
- Node.js 20 or 22 for the Teams client.

From PowerShell:

```powershell
git clone https://github.com/Jameschoong7/searchable-rag-copilot-system.git
Set-Location searchable-rag-copilot-system

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, company IT should approve
an appropriate execution policy. Do not disable enterprise security controls
globally merely to run this project.

Start FastAPI:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Start Streamlit in another PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run src/ui/app.py
```

## Teams Client Installation

Use [Node.js 20 or 22](https://nodejs.org/en/download). Node 24 is outside the
repository's declared engine range. Install Node in the environment where the
bot will run; for the primary WSL path, `node --version` must work inside WSL.
The primary path uses the Microsoft 365 Agents Toolkit in Visual Studio Code:

1. Install [Visual Studio Code](https://code.visualstudio.com/).
2. Install the [Microsoft 365 Agents Toolkit](https://learn.microsoft.com/microsoftteams/platform/toolkit/install-agents-toolkit)
   extension (`TeamsDevApp.ms-teams-vscode-extension`). If prompted in a WSL
   window, install it there too.
3. From the repository root, open the nested bot as its own VS Code workspace:

   ```bash
   code teams_bot/AgentsToolkitProjects/teams-chat-bot
   ```

   Do not use only `code .` from the repository root for this step. VS Code does
   not automatically load a nested folder's `.vscode/launch.json`.
4. Configure `RAG_API_BASE_URL=http://127.0.0.1:8000` in
   `env/.env.playground`.
5. Start the FastAPI backend separately.
6. Open **Run and Debug** (`Ctrl+Shift+D`).
7. Select **Debug in Microsoft 365 Agents Playground** and press `F5`.

The Toolkit prepares and opens the Playground automatically. No Microsoft 365
tenant is required for this local simulation.

### Command-Line Alternative

The project also carries Agents Playground as a development dependency, so a
clean clone can use:

```bash
cd teams_bot/AgentsToolkitProjects/teams-chat-bot
npm ci
npm run build
```

Start the bot process:

```bash
npm run dev:teamsfx:playground
```

In a second terminal, start the Playground client:

```bash
cd teams_bot/AgentsToolkitProjects/teams-chat-bot
npm run dev:teamsfx:launch-playground
```

Open the Playground URL printed by the second command. The URL on port `9239`
is Node DevTools for debugging and is not the chatbot interface. VS Code **Run
and Debug** -> **Debug in Microsoft 365 Agents Playground** performs these steps
automatically.

The Teams client is another frontend for the same FastAPI chat-job API. It does
not replace or host the backend.

## Clean Installation Acceptance Checklist

- [ ] Backend import prints `backend import ok`.
- [ ] `/health` returns HTTP 200 and the expected service name.
- [ ] Streamlit login loads without a traceback.
- [ ] System Admin can open Performance, AI Advisor, KB Management, Chat and
  Settings.
- [ ] General Employee cannot see administration controls.
- [ ] Local or Azure vector provider reports a nonzero indexed chunk count after
  indexing sample documents.
- [ ] A known query returns a cited answer.
- [ ] A restricted cross-department query returns a permission block.
- [ ] Graph device login and connector scan work if Graph is enabled.
- [ ] Teams client builds and can reach `/chat/jobs` if included in the test.

## Stop the System

Wait for active jobs to finish, then press `Ctrl+C` in Streamlit, FastAPI and
Ollama terminals. Running background work does not resume automatically after
FastAPI is stopped.
