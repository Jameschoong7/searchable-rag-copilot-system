# Installation Guide

This guide prepares a clean company test machine. WSL 2 is the primary path;
native Windows is an alternative for teams that cannot use WSL.

## Before You Start

Confirm the tester has:

- Windows 10 version 2004 or later, or Windows 11;
- permission to install WSL, Python packages and optional Microsoft tools;
- at least 16 GB RAM recommended for running Streamlit, FastAPI, local
  embeddings and Ollama together;
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
sudo apt install -y python3 python3-venv python3-pip git tesseract-ocr
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
git clone <COMPANY_REPOSITORY_URL> searchable-rag-copilot-system
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

Install Ollama using the approved installer for the test machine. Then:

```bash
ollama pull mistral
ollama serve
```

In a separate terminal:

```bash
curl http://127.0.0.1:11434/api/tags
```

If Azure OpenAI is active, Ollama is optional. It remains useful for a deliberate
offline fallback.

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
6. Open **KB Management** and check pending/indexed counts.
7. Run **Update Pending Documents** when documents are waiting.
8. Ask one known sample question and verify that sources are displayed.

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
git clone <COMPANY_REPOSITORY_URL> searchable-rag-copilot-system
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

Use Node.js 20 or 22. Node 24 is outside the repository's declared engine range.

```bash
cd teams_bot/AgentsToolkitProjects/teams-chat-bot
npm ci
npm run build
```

Configure `RAG_API_BASE_URL=http://127.0.0.1:8000` in
`env/.env.playground` for Agents Playground or `env/.env.local` for the local
Toolkit environment, then start Microsoft 365 Agents Playground:

```bash
npm run dev:teamsfx:playground
```

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
