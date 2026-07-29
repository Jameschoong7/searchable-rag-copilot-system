# Sample Knowledge Base Package

This directory defines the clean synthetic source structure supplied to company
testers. It must not contain personal Graph item identifiers, downloaded token
data, confidential company documents or runtime database files.

## Intended Layout

```text
sample_knowledge_base/
├── local_upload/
├── onedrive/
│   └── Enterprise Knowledge Base/
│       ├── ENGINEERING/
│       ├── HR/
│       ├── IT/
│       ├── OPERATIONS/
│       └── SECURITY/
├── onenote/
│   └── Enterprise Knowledge Base/
│       ├── ENGINEERING/
│       ├── HR/
│       ├── IT/
│       ├── OPERATIONS/
│       └── SECURITY/
├── SAMPLE_DATA_MANIFEST.md
└── labelled_queries.json
```

Git does not retain empty directories. Add department files/pages as the clean
sample set is curated.

## Source Rules

- `local_upload/`: original TXT/PDF/DOCX files for manual/ZIP workflows.
- `onedrive/`: files to copy into the same OneDrive folder hierarchy.
- `onenote/`: plain-text page bodies with title/section instructions; copy each
  body into the matching OneNote page.
- `labelled_queries.json`: a reviewed evaluation set aligned to this package.
- `SAMPLE_DATA_MANIFEST.md`: source identity, owner department, category, tags,
  ACL, expected questions and provenance.

The runtime `data/simulated/` directory is an ETL working area and can contain
connector-generated filenames. It should not be treated as the clean handover
source package.

## Minimum Recommended Corpus

Prepare at least:

- 3 documents per department across TXT, PDF and DOCX;
- one scanned PDF for OCR review;
- one document containing text plus an embedded image;
- one intentionally restricted policy;
- one document shared across departments;
- one version-update pair with changed facts;
- one OneDrive file per department;
- one OneNote page per department;
- one vague query expected to miss;
- one permission-block evaluation case.

All facts should be synthetic, internally consistent and easy to verify manually.
