# Troubleshooting Guide

Start with the failing layer instead of changing several providers at once.

## Diagnostic Order

1. Browser/Streamlit state.
2. FastAPI process and `/health`.
3. SQLite metadata/job state.
4. Active Settings versus `.env`.
5. Storage provider.
6. Vector provider and index freshness.
7. Embedding model.
8. LLM provider.
9. Microsoft Graph connector.
10. Teams client configuration.

## API Offline

Symptoms:

- header shows API Offline;
- chat submission says knowledge service unavailable;
- connector/settings calls fail immediately.

Checks:

```bash
curl -i http://127.0.0.1:8000/health
```

If it fails:

```bash
source .venv/bin/activate
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Read the deepest traceback in the FastAPI terminal. Common causes are wrong
working directory, inactive virtual environment, missing dependency or occupied
port.

## Streamlit Shows the Wrong Page or State

1. Navigate using the sidebar once more.
2. Wait for any connector/index job notice to update.
3. Refresh the browser.
4. Log out and sign in again.
5. Restart Streamlit only after confirming FastAPI remains healthy.

Do not delete SQLite/session data as a first response.

## Missing `department` Session Key

This should be handled by current session initialization. If it returns:

1. capture the exact login/logout sequence;
2. restart Streamlit;
3. verify the current source version includes session cleanup/initialization;
4. do not manually inject a department through browser controls.

## Upload Rejected

| Message/cause | Resolution |
|---|---|
| Missing file/title/ACL | Complete required fields. |
| Unsupported type | Use TXT, PDF or DOCX. |
| Empty file | Provide real content. |
| File over 20 MB | Reduce/split the synthetic test file. |
| Metadata/filename already exists | Use version replacement or a genuinely different source. |
| Permission denied | Use a role/department allowed to manage that source. |
| Storage error | Verify local permissions or Blob connection/container. |

An error must leave the current page usable. If navigation disappears, capture
the error and browser sequence before retrying.

## ZIP Staging Problems

- maximum compressed ZIP: 25 MB;
- maximum entries: 100;
- maximum declared expanded size: 100 MB;
- maximum individual supported file: 20 MB;
- directories themselves are skipped; supported files inside them are processed;
- unsafe traversal paths are rejected.

Review the per-file result rather than assuming the whole ZIP succeeded.

## Uploaded Document Is Not Searchable

Check the lifecycle in order:

1. metadata row exists;
2. status is approved/active;
3. chunk/index state is `pending_index` or indexed;
4. pending index job succeeded;
5. active vector backend matches the index that was updated;
6. querying user role and department are allowed;
7. filters do not exclude the file type/department;
8. similarity score passes the threshold.

Uploading does not automatically bypass review and indexing.

## Grounded Answer Says Source Was Not Found

If the source is listed among checked/retrieved candidates but the answer is not
grounded:

- inspect extracted text, especially OCR;
- ask a content-specific question rather than only a filename;
- verify chunking did not separate required context excessively;
- inspect retrieval scores;
- verify the active/latest source version;
- add a labelled query only after confirming the correct ground truth.

## Permission Block Missing or Unexpected

Check both dimensions:

- role must be included through the role hierarchy;
- department must be owner/allowed department.

A Project Manager role permission alone does not let every department's manager
read the document. System Admin remains globally privileged.

## Ollama Failure

```bash
curl http://127.0.0.1:11434/api/tags
ollama list
```

If Mistral is absent:

```bash
ollama pull mistral
```

Start `ollama serve` and verify `.env` plus SQLite Settings select `ollama`.

## Azure OpenAI Failure

Check without spending tokens:

- `LLM_BACKEND=azure_openai` active in Settings;
- endpoint ends with `/openai/v1`;
- API key is current;
- deployment name matches the deployed name exactly;
- Foundry deployment state is successful;
- subscription quota is available.

Use Settings **Test** for one real request only after these checks. HTTP 429 means
rate/quota pressure; retry later with backoff rather than repeatedly clicking.

## Azure Blob Failure

Check:

- storage connection string was rotated/updated correctly;
- container exists and name matches exactly;
- container is private but the account key remains authorized;
- FastAPI restarted after `.env` change;
- system clock/network is healthy;
- the connection string contains no accidental quotes/newlines.

Never paste the connection string into an issue report.

## Azure AI Search Failure

Check:

- endpoint format;
- admin key rather than query key;
- index name;
- Free-tier storage capacity;
- local embedding model availability;
- Settings active/pending provider state;
- whether a rebuild was interrupted after deleting the old index.

Azure Search counts can be eventually consistent. Refresh after a short delay,
but investigate a count that remains wrong after the job has succeeded.

## Graph 401 or Expired Token

Normal recovery:

1. ensure `GRAPH_ACCESS_TOKEN` is absent/commented;
2. run `python scripts/graph_device_login.py`;
3. verify `get_current_graph_user()`;
4. restart FastAPI;
5. scan again.

If silent refresh fails repeatedly, remove the local token cache only after
confirming the account can sign in again:

```text
data/auth/graph_token_cache.bin
```

Do not delete it during an active connector job.

## Graph Explorer Works but the Application Does Not

Graph Explorer uses its own Microsoft application, token, account and tenant.
Compare:

- `/me` identity;
- app registration account type;
- `GRAPH_AUTHORITY`;
- `GRAPH_CLIENT_ID`;
- delegated scopes;
- OneDrive/OneNote licensing in the signed-in tenant.

## OneDrive or OneNote Returns Nothing

- confirm exact root path/notebook name;
- verify files/pages are beneath the configured scope;
- verify supported file type and nonempty content;
- verify Graph account owns/can access that content;
- note that connector pagination is not fully implemented, so very large source
  collections can be incomplete.

## Connector Stage/Refresh Appears Stuck

1. Review sidebar active-job notice.
2. Check FastAPI terminal.
3. Poll the job only while FastAPI is running.
4. Do not submit repeated duplicate jobs.
5. After failure, rescan and select only the intended item.

The job record persists, but running Python background work does not survive a
backend restart.

## Teams Client Cannot Reach RAG

1. Verify FastAPI `/health`.
2. Check `RAG_API_BASE_URL=http://127.0.0.1:8000` in the Teams local environment.
3. Build with Node 20 or 22:

```bash
cd teams_bot/AgentsToolkitProjects/teams-chat-bot
npm ci
npm run build
```

4. Restart Agents Playground after environment changes.
5. Remember that `127.0.0.1` refers to the machine/process environment hosting
   the Teams client.

## SQLite Locked or Inconsistent State

Stop Streamlit and FastAPI before making a backup. Do not delete the database.
The current SQLite setup is single-machine/small-UAT oriented and has not been
proved for multi-worker writes.

Escalate with:

- exact operation and timestamp;
- active processes;
- database backup;
- job ID/status;
- sanitized traceback.

## Safe Support Bundle

Include:

- Git commit/version;
- operating system/WSL version;
- Python/Node versions;
- active provider names without endpoints/keys;
- sanitized error message;
- job ID and status;
- steps to reproduce.

Exclude:

- `.env`;
- connection strings and API keys;
- Graph access/token cache;
- personal OneDrive item IDs;
- confidential query/answer/document content.
