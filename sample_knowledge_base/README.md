# Sample Knowledge Base Package

This directory is the clean synthetic source package supplied to company
testers. It contains ready-to-use manual upload files, a department-scoped
OneDrive folder tree and OneNote page bodies. It must not contain personal Graph
item identifiers, downloaded token data, confidential company documents or
runtime database files.

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
└── generated_manifest.csv
```

Current package contents:

- 3 focused live-demo assets;
- 40 generated OneDrive DOCX/PDF files plus the focused IT demo PDF;
- 20 generated OneNote Markdown pages plus the focused Operations demo page;
- 65 physical files recorded in `generated_manifest.csv`;
- all five configured departments represented.

The focused OneDrive PDF and OneNote text are intentionally present twice: the
`local_upload/` copies are transfer/fallback source files, while the copies in
the connector trees show their intended destination. The manifest records both
physical copies so a tester can account for every supplied file.

## Source Rules

- `local_upload/`: original TXT/PDF/DOCX files for manual/ZIP workflows.
- `onedrive/`: files to copy into the same OneDrive folder hierarchy.
- `onenote/`: plain-text page bodies with title/section instructions; copy each
  body into the matching OneNote page.
- `generated_manifest.csv`: generated source inventory and connector placement.
- `SAMPLE_DATA_MANIFEST.md`: governance/expected-question template for the
  selected UAT subset.

The runtime `data/simulated/` directory is an ETL working area and can contain
connector-generated filenames. It should not be treated as the clean handover
source package.

The runtime evaluation file is `data/evaluation/labelled_queries.json`. Do not
copy its current connector-generated filenames blindly into a new environment.
After the company stages and indexes this package, update expected sources or
stable source IDs using the [Evaluation Guide](../docs/EVALUATION_GUIDE.md).

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
