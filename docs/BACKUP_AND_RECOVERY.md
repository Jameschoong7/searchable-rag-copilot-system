# Backup and Recovery

There is no automated backup tool. Use this controlled procedure for UAT.

## Data Inventory

| Data | Location | Importance |
|---|---|---|
| Governance/users/settings/jobs/chat/usage | `data/metadata/document_metadata.db` | Primary local system state. |
| Query logs | `data/logs/query_logs.db` | Operational/evaluation evidence. |
| Local/ETL source copies | `data/simulated/` | Required for local rebuild and audit. |
| Labelled tests/results/benchmarks | `data/evaluation/` | Retrieval-quality evidence. |
| Chroma index | `data/chroma_db/` | Rebuildable local vector state. |
| Azure Blob objects | Company Storage account | Durable hybrid source objects. |
| Azure AI Search index | Company Search service | Rebuildable retrieval index. |
| Graph token cache | `data/auth/graph_token_cache.bin` | Sensitive, replaceable sign-in state. |
| `.env` | Repository root, local only | Sensitive configuration. |

## Consistent Local Backup

1. Stop accepting new requests.
2. Wait for all background jobs to finish.
3. Stop Streamlit and FastAPI.
4. Create an access-controlled backup directory outside Git.
5. Copy:

```bash
BACKUP_DIR="$HOME/rag-copilot-backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp data/metadata/document_metadata.db "$BACKUP_DIR/"
cp data/logs/query_logs.db "$BACKUP_DIR/"
cp -a data/simulated "$BACKUP_DIR/"
cp -a data/evaluation "$BACKUP_DIR/"
cp -a data/chroma_db "$BACKUP_DIR/" 2>/dev/null || true
```

6. Record Git commit, active providers and timestamp in a text note without
   secrets.
7. Protect/encrypt the backup according to company policy.
8. Restart services.

Do not add normal copies of `.env` or Graph token cache to the backup archive.
Store approved secrets separately. The token cache can be recreated by login.

## Azure Backup Considerations

For hybrid mode:

- enable appropriate Blob soft delete/versioning according to company policy;
- document container/account/resource group;
- export or preserve source files, not only Azure AI Search vectors;
- treat Azure AI Search as rebuildable from approved metadata and source content;
- record index schema/name and active embedding model;
- use Azure-native export/backup tools approved by company IT where required.

## Restore Procedure

1. Stop Streamlit and FastAPI.
2. Back up the broken/current state before overwriting it.
3. Restore both SQLite databases and approved source/evaluation directories from
   the same consistent backup point.
4. Restore Chroma only if using the matching source/metadata/model state.
5. Do not restore someone else's Graph token cache; run device login instead.
6. Restore `.env` through the approved secret channel.
7. Start FastAPI and verify `/health`.
8. Start Streamlit and inspect Settings/metadata counts.
9. Rebuild the active vector index if it does not match restored metadata.
10. Run known-answer, permission-block and labelled evaluation checks.

## Recovery Scenarios

### Azure Search index lost

Use SQLite active metadata plus approved local/Blob source content to run a
controlled full rebuild. Confirm embedding model/dimension before rebuilding.

### Chroma index corrupt

Stop writers, preserve the corrupt folder for diagnosis, then rebuild from active
metadata/source copies into a clean target. Do not delete first without backup.

### SQLite database locked

Stop multiple writers, take a filesystem copy only after processes stop, and
investigate concurrency. Do not repeatedly kill/restart during a write job.

### Graph cache invalid

Remove the cache only after services/jobs stop, then run device login. Existing
indexed documents remain available; only new connector access is affected.

### Secret exposed

Rotate the provider credential; restoring an older `.env` containing the exposed
secret does not recover security.

## Backup Verification

A backup is not complete until a test restore verifies:

- databases open without corruption;
- users/metadata/settings are present;
- source files are readable;
- active version relationships remain intact;
- target vector index can be rebuilt;
- known queries and ACL checks pass.
