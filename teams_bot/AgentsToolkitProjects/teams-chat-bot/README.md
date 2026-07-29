# Searchable RAG Copilot Teams Client

This Microsoft 365 Agents Toolkit project is the standalone Teams-style chat
client for Searchable RAG Copilot. It calls the same FastAPI `/chat/jobs` backend
used by Streamlit.

## Prerequisites

- Node.js 20 or 22;
- Microsoft 365 Agents Toolkit and Agents Playground;
- Searchable RAG Copilot FastAPI running;
- dependencies installed with `npm ci`.

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

## Run in Agents Playground

Start FastAPI from the repository root first:

```bash
source .venv/bin/activate
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

The simplest option is VS Code **Run and Debug** -> **Debug in Microsoft 365
Agents Playground**. That launch target prepares the Playground environment,
starts the bot, starts the Playground, and attaches the debugger.

For the equivalent command-line flow, use two terminals after installing and
building the project:

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

See the repository root [README](../../../README.md) and
[Security and Limitations](../../../docs/SECURITY_AND_LIMITATIONS.md).
