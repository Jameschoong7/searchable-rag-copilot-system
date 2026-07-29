# Administrator Operations

This runbook covers routine System Admin operation after installation.

## Daily Start-Up

1. Start any required local dependency, such as Ollama.
2. Start FastAPI.
3. Verify `/health`.
4. Start Streamlit.
5. Sign in as System Admin.
6. Review API status and sidebar notices.
7. Open **Settings** and confirm active providers.
8. Review pending review/index counts before accepting user traffic.

## Upload Limits

| Workflow | Limit |
|---|---:|
| Single TXT/PDF/DOCX | 20 MB |
| ZIP upload | 25 MB compressed |
| ZIP file entries | 100 |
| ZIP declared expanded size | 100 MB |
| Individual file inside ZIP | 20 MB |

Empty, unsupported, path-unsafe, oversized and duplicate entries are rejected or
reported. ZIP staging does not immediately make documents searchable.

## Manual Upload

1. Open **KB Management → Upload → Upload New Document**.
2. Select TXT, PDF or DOCX.
3. Set trusted title, department, category and tags.
4. Select allowed roles and allowed departments.
5. Save the file and metadata.
6. Confirm the new row is **Pending Index**.
7. Run **Index Sync → Update Pending Documents**.
8. Verify the source through a known chat query.

System Admin is always retained in stored role ACLs. Selecting `General
Employee` expands the role hierarchy to include higher roles; department
permission remains independently required.

## Batch ZIP Staging

1. Prepare a ZIP containing supported files.
2. Prefer department directories such as `HR/` or `IT/` to support inference.
3. Open **Batch ZIP Upload**.
4. Upload and start staging.
5. Review staged, skipped and failed rows.
6. Open **Review Queue** for each staged record.
7. Correct inferred metadata and ACL.
8. Approve trusted items.
9. Update pending documents in one index job.

Batch ZIP is a staging convenience, not an ACL shortcut. Every accepted file
must pass metadata review.

## Review Queue

For each pending record:

1. Confirm the title represents the document.
2. Confirm department ownership.
3. Set a useful category and tags.
4. Select the minimum necessary roles.
5. Select the minimum necessary departments.
6. Review file/source/storage metadata.
7. Approve for indexing, or explicitly reject.

Rejected connector items may be staged again after correction. A duplicate
filename/source conflict should be resolved through version replacement rather
than creating a competing active record.

## OCR and Visual Status

PDF ingestion can classify:

- text extracted;
- visual content detected;
- OCR needed;
- OCR extracted;
- OCR review/error states.

Tesseract OCR is the local baseline. OCR text can contain character errors, so
administrators should review scanned-policy answers and evaluation cases. A PDF
remains one governed source record even when page-level extraction produces
multiple chunks.

## Index Sync

### Update pending documents

Use this for normal operation:

- new approved uploads;
- approved connector items;
- restored documents;
- changed versions;
- stale-vector cleanup scheduled by lifecycle actions.

The job removes vectors marked for replacement and adds current chunks. A tiny
progress fraction can appear visually empty when only one of hundreds of chunks
changes; inspect numeric details rather than the bar alone.

### Full rebuild

Use only when:

- creating the first index;
- deliberately changing vector provider;
- changing embedding model/schema;
- recovering from verified full-index inconsistency.

Azure Search rebuild deletes and recreates the configured index. Schedule a
maintenance window and back up metadata/source state first.

### Job behavior

Streamlit submits a backend job and polls its ID. UI tab changes and reruns do
not cancel a submitted job while FastAPI remains running. However, the Python
background task itself is not durable across a backend restart.

## Document Versioning

Each logical source should have:

- stable source identity;
- content hash;
- increasing version number;
- active/latest state;
- previous-version link;
- archived older versions.

### Replace manually uploaded content

Use **Upload New Version**, not Upload New Document with the same filename. After
successful backend staging, run the pending index update.

### Refresh OneDrive/OneNote content

Scan first to obtain current source information, then select the indexed item in
the refresh list. If its content hash changed, the backend creates the next
version. Run the pending index update to activate it.

### Same content

If content hash is unchanged, the connector should report no content change
rather than creating another version.

## Archive and Restore

### Archive

1. Select the active document in Knowledge Library.
2. Open the archive/lifecycle control.
3. Tick explicit confirmation.
4. Submit archive.
5. Wait for the lifecycle job result.

Archived documents are removed from active retrieval and hidden from the normal
knowledge table.

### Restore

1. Expand **Archived Documents**.
2. Select an intentionally manually archived record.
3. Confirm restore.
4. Submit.
5. Run the pending index update.

The restore list should focus on manually archived documents rather than every
automatically archived historical version.

## OneDrive and OneNote

Connector operation is:

```text
scan -> select -> stage job -> review -> approve -> pending index -> index update
```

Refresh operation is:

```text
edit upstream item -> scan -> select indexed item -> refresh job
-> changed version pending -> index update
```

If authentication fails, restart with the sequence in
[Graph Connector Setup](GRAPH_CONNECTOR_SETUP.md). A failed scan must not delete
or alter already indexed connector documents.

## Knowledge Library

Use filters to find documents by department, category, source, visual status and
index state. Select a document to review:

- identity and current version;
- title, department, category and tags;
- role and department ACL;
- source type and path;
- storage backend/URI;
- visual extraction status;
- active/index status.

Metadata ACL edits apply to retrieval immediately because ACL checks read SQLite
before vector search. Content changes still require reindexing.

## Performance and AI Advisor

Review:

- live answer outcomes and latency;
- connector/index jobs;
- version changes;
- labelled evaluation results;
- benchmark misses;
- AI Advisor recommendations.

AI Advisor recommendations are triage aids, not automatic changes. Verify the
underlying query, source and ACL before updating documents or permissions.

## Settings Change Control

Before changing providers:

1. Record current Settings.
2. Verify target credentials and service health.
3. Back up SQLite and source documents.
4. Confirm target index cost/quota.
5. Save the requested setting.
6. Complete required rebuild.
7. Run known-answer and permission-block tests.
8. Run labelled evaluation.
9. Roll back intentionally if acceptance fails.

Do not assume local and Azure vector indexes are synchronized.

## End-of-Day Checklist

- [ ] No active connector/index/chat jobs remain.
- [ ] Failed jobs have an owner and recorded follow-up.
- [ ] Pending review/index counts are understood.
- [ ] Azure usage/cost is within the UAT budget.
- [ ] No secrets appeared in logs or screenshots.
- [ ] Required backups completed before risky changes.
- [ ] Testers signed out before services stop.
