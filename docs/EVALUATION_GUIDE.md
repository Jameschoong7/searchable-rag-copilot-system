# Retrieval Evaluation Guide

Retrieval evaluation verifies whether the expected governed source appears in
the accepted Top-K chunks for a curated question and user scope.

It is separate from:

- answer wording quality;
- user satisfaction;
- live query outcome classification;
- LLM factual knowledge;
- latency/load testing.

## Files

| File | Purpose |
|---|---|
| `data/evaluation/labelled_queries.json` | Admin-reviewed test definitions. |
| `data/evaluation/retrieval_eval_results.json` | Latest generated results. |
| `src/evaluation/retrieval_eval.py` | Evaluation runner and summary logic. |

The results file is generated evidence. Do not hand-edit it to improve accuracy.

## Labelled Query Schema

Example expected-source hit:

```json
{
  "query_id": "Q-061",
  "suite": "hr_policy",
  "question": "When must the equipment allowance claim be submitted?",
  "role": "General Employee",
  "department": "HR",
  "department_filter": null,
  "file_type_filter": "DOCX",
  "expected_behavior": "hit",
  "expected_source": "hybrid_work_equipment_allowance.docx",
  "expected_source_document_id": null
}
```

Prefer a stable logical source ID when filenames change across versions:

```json
"expected_source": null,
"expected_source_document_id": "DOC-HR-EQUIPMENT-001"
```

Example intentionally vague expected miss:

```json
{
  "query_id": "Q-062",
  "suite": "guardrail",
  "question": "policy",
  "role": "System Admin",
  "department": "IT",
  "department_filter": "All",
  "file_type_filter": null,
  "expected_behavior": "miss",
  "expected_source": null
}
```

Example expected permission block:

```json
{
  "query_id": "Q-063",
  "suite": "acl",
  "question": "What are the restricted IT password requirements?",
  "role": "General Employee",
  "department": "HR",
  "department_filter": null,
  "file_type_filter": null,
  "expected_behavior": "permission_block",
  "expected_source": null
}
```

## Field Rules

| Field | Rule |
|---|---|
| `query_id` | Unique and stable. Do not reuse IDs. |
| `suite` | Groups related smoke, ACL, connector, OCR or policy tests. |
| `question` | Natural wording that a real tester might use. |
| `role` | One supported exact role label. |
| `department` | One supported exact department label. |
| `department_filter` | `All`, department value or `null`, consistent with role. |
| `file_type_filter` | `TXT`, `PDF`, `DOCX` or `null`. |
| `expected_behavior` | `hit`, `miss` or `permission_block`. |
| `expected_source` | Exact active filename when filename matching is appropriate. |
| `expected_source_document_id` | Stable logical source ID for versioned documents. |

Ground truth must be reviewed against the actual source content and ACL. Do not
generate expected sources from whatever the current retriever happens to return.

## Add a New Test

1. Confirm the source document is approved, active and indexed.
2. Read the source and define one answerable question.
3. Decide the correct role, department and optional filters.
4. Decide whether the expected behavior is hit, miss or permission block.
5. Use stable source ID for versioned connector/manual documents where possible.
6. Add a unique JSON object to `labelled_queries.json`.
7. Validate JSON syntax:

```bash
python -m json.tool data/evaluation/labelled_queries.json >/dev/null
```

8. Run the evaluation.
9. Review the exact failed rows and scores.
10. Commit the labelled test change separately from generated runtime noise when
    it represents accepted ground truth.

## Run Evaluation

Activate the environment and verify the intended Settings/index first:

```bash
source .venv/bin/activate
python -m src.evaluation.retrieval_eval
```

The command uses the active vector backend and can call the configured pipeline
for permission-block cases. It can incur Azure operations/model cost depending
on the cases and active providers.

The runner compares thresholds `0.25` and `0.30` in addition to writing the
latest result. Do not change the production threshold solely because one run has
a higher headline percentage.

## Metrics

```text
Top-K Accuracy = correct labelled cases / total labelled cases
Miss Rate      = failed labelled cases / total labelled cases
```

A hit means the expected source/source ID appeared within accepted Top-K chunks,
or the expected miss/permission behavior occurred. It does not prove that every
sentence in the generated answer is ideal.

## Interpret Misses

### Expected source absent

Check:

- document active/latest state;
- vector index freshness;
- role and department ACL;
- selected file/department filter;
- chunk boundaries and source metadata;
- similarity scores and threshold;
- duplicate/competing source content;
- expected filename versus stable source ID.

### Expected miss retrieved something

This is a false positive: one or more chunks passed the threshold for an
intentionally unsupported/vague query. It is not an ACL leak unless the accepted
chunk was unauthorized. Review the score and query specificity before raising
the threshold globally.

### Permission block failed

Check that a relevant restricted source exists and that the test account lacks
either role or department permission. A normal Not Found can occur when no
restricted candidate is detected at all.

## Why Accuracy Should Not Be Forced to 100%

Real retrieval systems face ambiguous wording, overlapping documents, stale
indexes, threshold trade-offs and changing corpora. Keeping a defensible miss in
the benchmark can demonstrate honest measurement and continuous improvement.

The administrator's responsibility is to:

- add representative labelled questions over time;
- investigate each miss;
- distinguish threshold, metadata, ACL, content and chunking causes;
- record accepted improvements;
- avoid rewriting ground truth merely to match system output.

## Evaluation Change Checklist

- [ ] New query reflects a real knowledge need.
- [ ] Expected source is manually verified.
- [ ] ACL identity is realistic.
- [ ] Active index contains the intended version.
- [ ] JSON validates.
- [ ] Evaluation was run against the intended backend.
- [ ] Misses were reviewed, not hidden.
- [ ] Result records backend/model/index context in the accompanying test note.
