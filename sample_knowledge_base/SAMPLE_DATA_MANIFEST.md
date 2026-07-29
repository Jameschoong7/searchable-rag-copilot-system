# Sample Data Manifest

Complete one row for every clean sample source before company UAT.

| Source ID | Filename/page title | Source channel | Department | Category | Tags | Allowed roles | Allowed departments | Version | Expected question | Expected fact/source | Provenance |
|---|---|---|---|---|---|---|---|---:|---|---|---|
| `SAMPLE-HR-001` | `<add file>` | Manual | HR | Employee Benefits | `<tags>` | System Admin; Project Manager; General Employee | HR | 1 | `<question>` | `<fact>` | Synthetic |
| `SAMPLE-IT-001` | `<add file>` | OneDrive | IT | IT Operations | `<tags>` | System Admin; Project Manager; General Employee | IT | 1 | `<question>` | `<fact>` | Synthetic |
| `SAMPLE-OPS-001` | `<add page>` | OneNote | Operations | Handover | `<tags>` | System Admin; Project Manager; General Employee | Operations | 1 | `<question>` | `<fact>` | Synthetic |

## Review Checklist

- [ ] Source contains no confidential or personal data.
- [ ] Department and ACL match the intended test.
- [ ] File/page opens and extraction is readable.
- [ ] Expected fact is explicitly present.
- [ ] Version/update facts are not contradictory unless intentionally testing
  stale-version exclusion.
- [ ] Labelled query uses stable source ID where versioning is involved.
- [ ] OneDrive/OneNote placement matches connector inference structure.
