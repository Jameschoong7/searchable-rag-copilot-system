# Company UAT Guide

This guide defines a controlled evaluation, not unrestricted production use.

## Recommended Pilot

- 3 to 5 named testers;
- 5 to 7 working days;
- synthetic, public or explicitly approved non-confidential documents;
- approximately 20 to 50 questions per tester per day;
- one designated System Admin/operator;
- company-owned Azure resources and credentials for the hybrid profile;
- daily cost, error and feedback review.

## Roles to Assign

| Participant | Demo account | Purpose |
|---|---|---|
| Operator | `admin_jc` | Ingestion, settings, indexing and global monitoring. |
| Department reviewer | `pm_<department>` | Department-scoped search and KB review. |
| Employee tester | `employee_<department>` | Simple department-scoped chat. |

Do not let external testers call FastAPI directly or use Teams `/use-admin`
profile switching as though it were real authentication.

## Pre-UAT Checklist

- [ ] Company approves test data and retention period.
- [ ] Exposed/personal Azure keys have been rotated.
- [ ] Company-owned `.env` exists only on the test machine.
- [ ] Blob container is private.
- [ ] Azure AI Search is Free tier and below 50 MB.
- [ ] Foundry budget alerts and owner contacts are configured.
- [ ] Graph connector uses a dedicated approved account/root/notebook.
- [ ] Backend and portal are bound to an approved network interface.
- [ ] Seeded passwords are shared only with named testers.
- [ ] Sample documents are approved and indexed.
- [ ] Known-answer, Not Found and permission-block checks pass.
- [ ] Backup of clean pre-UAT SQLite/data state exists.

## Tester Scenarios

### General Employee

1. Sign in and verify role/department.
2. Ask an answerable own-department question.
3. Inspect citations.
4. Ask a contextual follow-up with memory enabled.
5. Ask an unsupported question and verify grounded refusal.
6. Ask a known restricted cross-department question and verify permission block.
7. Review KB Status and confirm restricted documents are absent.

### Project Manager

1. Repeat cited chat within own department.
2. Use a file-type filter.
3. Confirm another department cannot be freely selected.
4. Review authorized Knowledge Library records.
5. Upload one own-department test document.
6. Confirm it remains Pending Index until admin processing.
7. Review department-scoped Performance and AI Advisor records.

### System Admin

1. Review global Performance and current architecture.
2. Upload, review and index one test document.
3. Scan/stage one connector item if Graph is included.
4. Replace a document and confirm old content is excluded.
5. Archive/restore one synthetic record.
6. Run labelled retrieval evaluation outside peak testing.
7. Review AI Advisor and usage records.

## Feedback Form

For every reported issue, collect:

- tester account/role/department;
- timestamp;
- page/client;
- exact question or action;
- selected filters;
- expected outcome;
- actual outcome/status;
- displayed sources without secret URLs;
- screenshot with credentials/endpoints hidden;
- whether retry changed the result.

Do not collect Azure keys, Graph tokens or `.env` screenshots.

## Daily Operator Review

1. Check failed/long-running backend jobs.
2. Review Not Found and permission-block counts.
3. Review user feedback and candidate knowledge gaps.
4. Check pending review/index queues.
5. Review Azure Cost Analysis and Foundry token usage.
6. Confirm Search storage remains within Free-tier capacity.
7. Back up accepted metadata/document changes.

## Acceptance Criteria

The controlled UAT is successful when:

- permitted users obtain grounded cited answers for agreed sample scenarios;
- restricted users cannot retrieve restricted source content/path;
- manual and selected connector ingestion follow review-before-index governance;
- updated versions replace old retrieval content;
- labelled evaluation runs reproducibly with explained misses;
- errors are visible and recoverable without exposing secrets;
- Azure usage remains within the agreed budget;
- company testers can follow the provided setup/manual without developer-only
  knowledge.

## End of UAT

1. Stop accepting uploads/questions.
2. Wait for jobs to finish.
3. Export agreed feedback and evaluation evidence.
4. Back up or securely delete UAT metadata/documents according to policy.
5. Revoke Graph consent or delete token cache if no longer required.
6. Rotate temporary Azure keys if they were distributed to operators.
7. Remove test-machine `.env` when decommissioning.
8. Record limitations and production-hardening decisions.
