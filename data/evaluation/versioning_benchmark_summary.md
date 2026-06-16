# Versioning And Active-Aware Rebuild Benchmark

## Purpose

This benchmark addresses supervisor feedback about document updates, old document versions, vector database growth, and the risk of RAG answers using outdated information.

## Scenario

The password policy document was updated from version 1 to version 2.

- Version 1 file: `IT_Policy_Password.txt`
- Version 1 expiry rule: passwords expire every 90 days
- Version 2 file: `IT_Policy_Password_v2.txt`
- Version 2 expiry rule: passwords expire every 60 days

The metadata system archived version 1 and made version 2 active while preserving the same source document identity.

## Before Active-Aware Rebuild

After creating version 2:

| Metric | Value |
| --- | ---: |
| Active metadata records | 16 |
| Physical source files | 17 |
| Chroma vector count | 90 |
| Chroma DB size | 1.54 MB |

Observation: the old version was logically inactive, but full folder-based rebuild still indexed both version 1 and version 2.

## After Active-Aware Rebuild

After updating ETL to index only active metadata filenames:

| Metric | Value |
| --- | ---: |
| Active metadata records | 16 |
| Physical source files | 17 |
| Chroma vector count | 84 |
| Chroma DB size | 1.49 MB |

Observation: version 1 can remain on disk for audit/history, but its vectors are removed from the active Chroma index.

## Correctness Check

Query:

```text
When do passwords expire?
```

Result:

```text
Passwords expire every 60 days.
```

Source:

```text
data/simulated/IT_Policy_Password_v2.txt
```

```markdown
## Full Rebuild vs Single-Document Update Benchmark

| Benchmark | Scope | Chunks Processed | Deleted Vectors | Final Vector Count | Elapsed Time |
| --- | --- | ---: | ---: | ---: | ---: |
| Active-aware full rebuild | All active documents | 84 | Recreated index | 84 | 10.678s |
| Single-document update | `IT_Policy_Password_v2.txt` only | 6 | 6 | 84 | 6.213s |

## Benchmark Interpretation

The active-aware full rebuild is useful for complete index cleanup because it reconstructs Chroma from the current active metadata set. It guarantees archived documents are removed from the active vector index, but it reprocesses all active documents.

The single-document update path is more efficient for normal document changes. It deletes vectors for the changed source file, re-embeds only that file, and keeps the final vector count stable. In this small local corpus, it processed 6 chunks instead of 84 chunks and completed faster than the full rebuild.

This supports the supervisor feedback that incremental update can reduce unnecessary re-embedding work, control vector database growth, and lower future Azure embedding cost when the corpus becomes larger.
```

## Conclusion

The project now supports logical document versioning and active-aware full rebuild. This reduces the risk of outdated answers and prevents archived document versions from continuing to grow the active vector index.

The next improvement is updated-document-only reindexing, so the system does not need to rebuild the entire Chroma database after one document changes.
```

Then commit:

```bash
git add data/evaluation/versioning_benchmark_summary.md PROJECT_MEMORY.md
git commit -m "Document versioning benchmark result"
```

After that, we can move to the next harder part: **update-only indexing**.
