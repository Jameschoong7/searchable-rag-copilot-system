# Company Handover Manifest

## Release Identity

| Field | Value |
|---|---|
| Project | Searchable RAG Copilot |
| Academic project | UOW Malaysia industry final-year project with Centific Malaysia |
| Handover date | 31 July 2026 |
| Repository | <https://github.com/Jameschoong7/searchable-rag-copilot-system> |
| Release tag | `v1.0-company-handover` |
| Tagged ZIP filename | `searchable-rag-copilot-system-v1.0-company-handover.zip` |
| Application baseline commit | `28b2970f316785524c53a1a283021c2bb5f338fc` |
| Tagged release commit | `$Format:%H$` |
| Handover contact | Jun Cheng Choong (James), GitHub `@Jameschoong7` |

The tagged release commit token is expanded to the exact commit SHA when the
official ZIP is generated with `git archive`. In a normal Git checkout, the
`v1.0-company-handover` tag target shown by Git/GitHub is authoritative.

## Delivery Classification

This release is a fixed source handover for:

- academic assessment and supervisor review;
- controlled company-side user acceptance testing;
- synthetic, public or explicitly approved non-confidential documents;
- local or cost-controlled hybrid Azure evaluation described in the guides.

It is not an unrestricted production deployment. Do not expose FastAPI or the
demonstration clients directly to untrusted users or the public internet.

## Main Documentation

| Purpose | Location |
|---|---|
| Entry point and setup paths | `README.md` |
| Installation and startup | `docs/INSTALLATION.md` |
| Providers, prompts and demonstration accounts | `docs/CONFIGURATION.md` |
| Azure Blob, AI Search and Foundry setup | `docs/AZURE_SETUP.md` |
| Graph, OneDrive and OneNote setup | `docs/GRAPH_CONNECTOR_SETUP.md` |
| Role-based portal use | `docs/USER_MANUAL.md` |
| Administration and maintenance | `docs/ADMIN_OPERATIONS.md` |
| Retrieval evaluation | `docs/EVALUATION_GUIDE.md` |
| Controlled company test | `docs/COMPANY_UAT_GUIDE.md` |
| Troubleshooting | `docs/TROUBLESHOOTING.md` |
| Security and limitations | `docs/SECURITY_AND_LIMITATIONS.md` |
| Backup and recovery | `docs/BACKUP_AND_RECOVERY.md` |
| Architecture and data flow | `docs/ARCHITECTURE.md` |
| Standalone Teams-style client | `teams_bot/AgentsToolkitProjects/teams-chat-bot/README.md` |

## Included Sample Knowledge Base

The clean synthetic handover corpus is under `sample_knowledge_base/`:

- `local_upload/`: focused manual/connector transfer files;
- `onedrive/Enterprise Knowledge Base/`: five department folder trees;
- `onenote/Enterprise Knowledge Base/`: five department page trees;
- `generated_manifest.csv`: inventory of all 65 physical sample files;
- `SAMPLE_DATA_MANIFEST.md`: selected test-governance template.

The runtime `data/simulated/` folder contains the tracked seed corpus and is also
used as the local ETL working area. Connector downloads and later uploads are
runtime state and are not all part of the clean handover corpus.

## Intentionally Excluded

The Git tag and tagged ZIP do not include ignored/untracked local state such as:

- `.env`, Azure keys, connection strings and access tokens;
- `.venv/`, Python caches and Node `node_modules/`;
- Graph/MSAL token caches under `data/auth/`;
- SQLite runtime databases under `data/metadata/`;
- ChromaDB runtime indexes under `data/chroma_db/`;
- query/usage logs under `data/logs/`;
- generated retrieval result output;
- local connector downloads matching `data/simulated/onedrive_*` and
  `data/simulated/onenote_*`;
- local planning/audit files such as `PROJECT_MEMORY.md`, `PLANS.md`, `.codex/`
  and `docs/a5_evidence/`;
- personal Azure/Graph identifiers and confidential company documents.

After installation, the operator creates a private `.env`, initializes local
runtime databases and builds the selected vector index by following the
installation guide.

## Known Limitations

- Controlled-UAT demonstration accounts are not production Entra identity.
- FastAPI currently trusts client-supplied user/role/department fields; it must
  not be directly exposed to untrusted callers.
- Teams `/use-*` profiles are simulations, not Microsoft Teams identity mapping.
- SQLite and in-process background jobs are intended for a single controlled
  test node, not multi-instance/high-concurrency production operation.
- Production TLS, reverse proxy, secret manager, private networking, rate
  limiting, malware scanning and durable external workers are not packaged.
- Microsoft Graph uses delegated read access and does not provide live
  SharePoint integration or enterprise SharePoint ACL mirroring in this release.
- OCR, source-path citations and connector synchronization have the limitations
  recorded in `docs/SECURITY_AND_LIMITATIONS.md`.
- The Teams local development dependency chain retains the moderate advisories
  documented in the Teams and security guides.
- Azure AI Search Free and paid-model usage must remain within the company test
  budget and quota.

Before distribution, the operator must confirm that any Azure key exposed during
development has been rotated and that the handover `.env` contains only approved
company-owned credentials.

## Ownership And Permitted Use

No open-source licence is granted by this repository or handover manifest.
Third-party dependencies and generated tooling remain subject to their own
licence terms.

This package is provided only for the academic review and controlled company UAT
purposes described above. Project ownership, institutional intellectual-property
rights, commercial use, redistribution, sublicensing and use beyond the agreed
evaluation scope remain subject to the applicable university rules and any
written agreement among the student, university and company.

Before any broader internal, production, commercial or redistributed use, the
recipient must obtain written confirmation from the relevant rights holder(s).
This section records the handover boundary; it is not a substitute for legal or
institutional advice.

## Integrity Verification

Verify the tag and commit after cloning:

```bash
git fetch --tags origin
git show --no-patch --decorate v1.0-company-handover
git rev-list -n 1 v1.0-company-handover
```

Verify the separately supplied ZIP against its accompanying
`SHA256SUMS.txt`:

```bash
sha256sum -c SHA256SUMS.txt
```
