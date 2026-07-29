# Architecture

## System Context

```text
                         +----------------------+
                         | Streamlit Web Portal |
                         | admin/PM/employee    |
                         +----------+-----------+
                                    |
                                    | HTTP jobs and admin APIs
                                    v
+--------------------+    +---------+----------+    +----------------------+
| Teams-style Client |--->| FastAPI Shared     |--->| Configured LLM       |
| employee chat      |    | Backend Brain      |    | Ollama or Foundry    |
+--------------------+    +----+-----------+----+    +----------------------+
                              |           |
                         ACL/metadata     authorized retrieval
                              |           |
                              v           v
                        +-----+----+  +---+------------------+
                        | SQLite  |  | Chroma or Azure       |
                        | control |  | AI Search             |
                        +-----+----+  +----------------------+
                              |
                              v
                 +------------+----------------+
                 | Local files / Azure Blob /  |
                 | OneDrive / OneNote sources  |
                 +-----------------------------+
```

The Streamlit portal and Teams client are separate frontends. Neither owns the
RAG engine or governance state.

## Core Components

| Component | Responsibility |
|---|---|
| `src/ui/app.py` | Portal presentation, login session, forms, polling and role-specific navigation. |
| `src/api/main.py` | Shared HTTP contracts, backend validation and background jobs. |
| `src/rag/engine.py` | ACL-aware retrieval, prompting, answer generation and citations. |
| `src/etl/pipeline.py` | TXT/PDF/DOCX extraction, OCR, cleaning, chunking and embedding. |
| `src/metadata/repository.py` | SQLite document metadata/source-of-truth operations. |
| `src/vector/` | Chroma/Azure AI Search adapters and provider selection. |
| `src/storage/` | Local/Azure Blob document storage adapter. |
| `src/connectors/` | Microsoft Graph authentication and OneDrive/OneNote reads. |
| `src/evaluation/` | Labelled retrieval evaluation and index benchmarks. |
| `teams_bot/...` | Teams-style client over the shared chat-job API. |

## Query Flow

```text
User question
  -> client submits /chat/jobs
  -> backend creates/continues owned chat session
  -> optional bounded follow-up rewrite
  -> load trusted document metadata from SQLite
  -> enforce user role + department ACL
  -> apply optional search filters inside permitted scope
  -> vector similarity search
  -> normalize score and apply threshold
  -> accept up to Top-K authorized chunks
  -> build grounded prompt
  -> invoke configured LLM
  -> return answer + unique source citations
  -> store chat message, query outcome, job and usage records
```

Permissions are enforced before context reaches the LLM. The LLM does not decide
whether a document is allowed.

## Memory Flow

```text
session ID + new question
  -> load bounded recent messages owned by the same user
  -> rewrite ambiguous follow-up into standalone retrieval question
  -> run normal current ACL/filter/retrieval flow
  -> display/store original question and answer
```

Memory improves query clarity but does not carry forward authorization or source
permission.

## Ingestion Flow

```text
manual file / ZIP / OneDrive / OneNote
  -> backend receives or downloads source
  -> local or Azure Blob storage
  -> SQLite pending-review/pending-index metadata
  -> admin/PM scope validation
  -> trusted metadata and ACL review
  -> extraction/OCR
  -> cleaning and chunking
  -> local MiniLM embeddings
  -> Chroma or Azure AI Search
  -> indexed active/latest state
```

## Version Flow

```text
stable source ID
  -> compare content hash
  -> unchanged: skip new version
  -> changed: create version N+1
  -> archive/deactivate version N
  -> schedule old-vector removal + new-vector insertion
  -> retrieve active/latest only
```

Archived versions remain available for governance/audit but are excluded from
normal retrieval.

## Data Ownership

SQLite owns:

- document identity, metadata and ACL;
- active/version/archive state;
- users and runtime settings;
- jobs;
- chat sessions/messages;
- LLM usage;
- query logs.

Vector providers duplicate chunk metadata needed for filtered retrieval. They are
not the sole source of governance truth.

## Provider Selection

```text
STORAGE_BACKEND=local | azure_blob
VECTOR_BACKEND=chroma | azure_search
EMBEDDING_BACKEND=local
LLM_BACKEND=ollama | azure_openai
```

SQLite Settings can override environment defaults. Provider fallback is manual,
and inactive vector indexes can become stale.

## Trust Boundaries

- browser/Teams input is untrusted;
- Streamlit demo login provides UAT presentation identity only;
- FastAPI currently trusts asserted client identity and therefore must remain in
  a controlled environment;
- SQLite metadata is trusted for ACL decisions;
- retrieved content can contain untrusted document text and is constrained by
  grounded prompting;
- Azure/Graph keys and token caches are secrets outside Git;
- cloud providers are external dependencies with quota, availability and cost.

See [Security and Limitations](SECURITY_AND_LIMITATIONS.md) for production gaps.
