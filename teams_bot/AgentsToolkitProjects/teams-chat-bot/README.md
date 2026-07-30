# Searchable RAG Copilot Teams Client

This Microsoft 365 Agents Toolkit project is the standalone Teams-style chat
client for Searchable RAG Copilot. It calls the same FastAPI `/chat/jobs` backend
used by Streamlit.

## Prerequisites

- [Node.js 20 or 22](https://nodejs.org/en/download) in the environment where
  this bot runs;
- [Visual Studio Code](https://code.visualstudio.com/);
- [Microsoft 365 Agents Toolkit](https://learn.microsoft.com/microsoftteams/platform/toolkit/install-agents-toolkit)
  extension
  (`TeamsDevApp.ms-teams-vscode-extension`);
- Searchable RAG Copilot FastAPI running;
- `RAG_API_BASE_URL` configured in `env/.env.playground`.

## Configure the Backend URL

For Agents Playground, edit:

```text
env/.env.playground
```

For a locally provisioned Teams Toolkit environment, edit:

```text
env/.env.local
```

Set:

```dotenv
RAG_API_BASE_URL=http://127.0.0.1:8000
```

Do not place Azure OpenAI, Search, Blob or Graph credentials in this Teams
project. The Teams client calls FastAPI; the backend owns provider credentials.

## Primary: Run With Agents Toolkit

Start FastAPI from the repository root first:

```bash
source .venv/bin/activate
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Then, from the repository root, open this nested bot folder as its own VS Code
workspace:

```bash
code teams_bot/AgentsToolkitProjects/teams-chat-bot
```

Opening only the repository root with `code .` does not load this folder's
`.vscode/launch.json`. In the bot workspace:

1. Install the recommended **Microsoft 365 Agents Toolkit** extension when VS
   Code prompts, or search for extension ID
   `TeamsDevApp.ms-teams-vscode-extension`.
2. Open **Run and Debug** (`Ctrl+Shift+D`).
3. Select **Debug in Microsoft 365 Agents Playground**.
4. Press **Run** or `F5`.

The Toolkit checks prerequisites, installs/prepares Agents Playground, starts the
bot on port `3978`, opens Playground and attaches the debugger on port `9239`.
This local simulation does not require a Microsoft 365 tenant.

## Alternative: Run From Terminals

Agents Playground is included as a project development dependency, so `npm ci`
installs everything needed for this alternative. Use two terminals:

```bash
cd teams_bot/AgentsToolkitProjects/teams-chat-bot
npm ci
npm run build
npm run dev:teamsfx:playground
```

In a second terminal:

```bash
cd teams_bot/AgentsToolkitProjects/teams-chat-bot
npm run dev:teamsfx:launch-playground
```

Open the Microsoft 365 Agents Playground URL printed by the second command. Do
not use the Node DevTools URL on port `9239` as the chat interface; that port is
only for debugging the bot process. The welcome message is sent on the
`install.add` activity raised by Playground or Teams, so it is not expected in
Node DevTools or on the bot's raw HTTP endpoint.

## Demo Profiles

The current client uses conversation-scoped UAT profile commands:

- `/profile` shows the active profile;
- `/use-hr` selects the HR General Employee profile;
- `/use-it-manager` selects the IT Project Manager profile;
- `/use-admin` selects the System Admin profile.

Once changed, the profile remains active for later messages in the same
conversation because the client stores it by conversation ID.

These commands simulate access profiles. They are not Microsoft Teams identity
authentication and must not be enabled in an unrestricted or production client.

## Chat Job Flow

```text
Teams message
  -> POST FastAPI /chat/jobs
  -> receive job_id
  -> poll GET /admin/jobs/{job_id}
  -> send grounded answer and sources
```

The client polls once per second for up to 30 attempts. If the backend reports a
failed job or the polling window expires, the client sends a controlled error
message.

## Verification

1. Send `hi` and confirm the client asks for a knowledge request without citing
   irrelevant documents.
2. Run `/profile`.
3. Ask an HR policy question as `/use-hr` and inspect sources.
4. Switch to `/use-it-manager` and ask a restricted HR question.
5. Confirm the outcome appears in Streamlit Performance/AI Advisor logs because
   both clients use the shared backend logging path.

## Known Limits

- profile commands are simulated authentication;
- Teams does not currently pass the Streamlit persistent chat session ID, so
  cross-message memory behavior is more limited;
- source citations are text paths, not Adaptive Card document previews;
- production Teams registration/deployment is not included in this repository;
- FastAPI must remain running and reachable from the client process.
- `npm audit --omit=dev` currently reports four moderate advisories inherited
  from Microsoft `@microsoft/teams.dev@2.0.12`, which pins an older
  `@microsoft/teams.apps`/`uuid` chain. The latest compatible SDK still carries
  that pin; monitor Microsoft SDK updates before any production deployment.

See the repository root [README](../../../README.md) and
[Security and Limitations](../../../docs/SECURITY_AND_LIMITATIONS.md).
