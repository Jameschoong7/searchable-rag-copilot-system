# User Manual

This manual covers the implemented Streamlit portal for three controlled-UAT
roles. UI visibility and retrieval scope are bound to the signed-in demo account.

## Accounts and Access Model

All seeded UAT users initially use `password123`.

| Role | Accounts | Main scope |
|---|---|---|
| System Admin | `admin_jc` | All departments and administration functions. |
| Project Manager | `pm_<department>` | Own department plus documents explicitly shared through ACL. |
| General Employee | `employee_<department>` | Own department plus documents explicitly shared through ACL. |

Supported department suffixes are `it`, `hr`, `engineering`, `security` and
`operations`.

The current login is suitable only for controlled UAT. It is not company Entra
authentication.

## Sign In

1. Confirm the backend is running and the header shows **API Online**.
2. Enter the assigned username.
3. Enter the password.
4. Select **Sign In**.

Expected result:

- the sidebar shows username, role and department;
- System Admin and Project Manager start on **Performance**;
- General Employee starts on **KB Status**;
- navigation options differ by role.

If login fails, verify spelling and account state. There is no self-service
password reset or account administration screen.

## Global Navigation

| Page | System Admin | Project Manager | General Employee |
|---|:---:|:---:|:---:|
| Performance | Yes | Yes, department-scoped | No |
| AI Advisor | Yes | Yes, department-scoped | No |
| KB Management / KB Status | Yes | Yes | View-only status |
| Chat | Yes | Yes | Yes |
| Settings | Yes | No | No |

Use **Logout** to clear the current Streamlit session and return to the login
screen. Do not close the browser as a substitute on a shared computer.

## Chat

### Start a conversation

1. Open **Chat**.
2. Review the displayed account role, department and filter scope.
3. Leave **Use memory** on for contextual follow-ups, or switch it off to avoid
   the additional rewrite-model call.
4. Enter a specific knowledge question.
5. Select **Send**.
6. Wait while the chat job is queued and processed.

The backend stores the user question, applies the current ACL and filters,
retrieves authorized chunks, generates an answer and records the result.

### Understand filters

- System Admin can choose department and file type globally.
- Project Manager can filter file type but cannot expand beyond permitted
  department/ACL scope.
- General Employee sees locked scope information rather than editable filters.

Filters narrow an already-authorized set. They cannot grant access.

### Read answer states

| State | Meaning | What to do |
|---|---|---|
| Grounded answer | Authorized evidence passed retrieval and supported an answer. | Review the answer and sources. |
| Not Found | The permitted knowledge base did not support the request. | Rephrase with a clearer policy, process, department or task. |
| Insufficient Permission | Relevant restricted content exists outside the account's ACL. | Request access through the organization's normal process. |
| Connection/API error | The backend or provider was unavailable. | Ask an administrator to check system notices and provider health. |

### Review sources

Expand the source area beneath a grounded answer. A Top-K setting of five means
up to five chunks, not necessarily five different files. Several chunks can
belong to the same source, so the unique citation list can be shorter.

Current citations show source paths and extracted context. They do not provide a
clickable original-document preview or exact PDF page anchor in every case.

### Continue a conversation

With **Use memory** enabled:

1. Ask an initial grounded question.
2. Ask a follow-up such as `What about step 3?`.
3. The backend loads a bounded recent history and rewrites the follow-up into a
   standalone retrieval question.
4. The rewritten question still performs a fresh ACL-filtered retrieval.

Memory never reuses a previous source without retrieval and never bypasses the
current account scope.

### Manage conversations

- **New Chat** creates a new session ID and empty conversation.
- **Recent Conversations** lists sessions owned by the current user.
- Select a session and choose **Open** to reload its messages.
- **Clear Current** clears the current conversation state/start point; it is not
  a full administrator data-retention/delete function.
- **Good** and **Issue** record lightweight answer feedback where available.

### Suggested questions

Expand **Suggested Questions** and select an example to populate/submit a known
scenario. Suggested questions do not change permissions.

## KB Status: General Employee

1. Open **KB Status**.
2. Review counts for documents visible to the current role/department.
3. Use available knowledge-library filters.
4. Select a visible document to inspect its metadata and storage information.

General Employees cannot upload, stage, approve, edit, archive, restore or index
documents. Hidden documents and restricted source paths should not appear.

## KB Management: Project Manager

### Review authorized documents

1. Open **KB Management**.
2. Use department/category/source/visual/index filters.
3. Select a visible document.
4. Review overview and storage metadata.

Project Managers see own-department documents plus records shared to their role
and department. They do not gain global visibility merely because a document
allows the Project Manager role; the department ACL must also permit them.

### Upload a new document

1. Open **Upload → Upload New Document**.
2. Select a TXT, PDF or DOCX file no larger than 20 MB.
3. Confirm/edit the title.
4. Confirm the locked manager department.
5. Enter category and comma-separated tags.
6. Select allowed roles and the manager's permitted department.
7. Select **Save File + Metadata**.

The upload becomes **Pending Index**. A System Admin performs the index update.
Project Managers cannot assign another department or System Admin-only global
scope through this form.

### Upload a new version

1. Open **Upload → Upload New Version**.
2. Select an active manageable source.
3. Select the replacement TXT, PDF or DOCX.
4. Submit the replacement.

The backend creates the next version, links it to the stable source identity and
marks it pending. The old active version is archived so both versions do not
compete in retrieval. Search changes after the pending index update succeeds.

### Review queue

Project Manager review is limited to permitted department items. Confirm the
metadata and ACL before approval. Index execution remains a System Admin task.

## Performance: System Admin and Project Manager

### System overview

Review indexed-document state, query volume, latency, answer outcomes and
retrieval evaluation summaries. These metrics combine different evidence types:

- live query signals come from operational logs;
- Top-K Accuracy comes from labelled expected-source tests;
- index benchmarks compare full rebuild and incremental update behavior.

Do not describe operational query outcomes as labelled retrieval accuracy.

### Recent outcomes

Expand **Recent Outcomes** to review recent query status, latency and source
behavior. Project Manager data is scoped to that manager's department and
excludes System Admin activity even when the admin belongs to the same department.

### Benchmark misses

Expand **Benchmark Misses** to inspect labelled cases that did not meet their
expected behavior. A miss can mean:

- expected source absent from Top-K;
- an intentionally vague query accepted an unrelated chunk;
- expected permission block not returned;
- stale metadata/index state;
- threshold or source-label mismatch.

An administrator should investigate before changing the threshold.

## AI Advisor

AI Advisor turns weak live query outcomes into operational improvement actions.

1. Open **AI Advisor**.
2. Select an issue category such as knowledge gap, permission/ACL or service
   reliability.
3. Review the issue, priority, reason, responsible role and recommended action.
4. Expand raw records only when detailed evidence is needed.
5. Optionally generate an action plan for one selected recommendation.

The first classification is deterministic. Generating an action plan invokes
the configured LLM and can consume tokens. AI Advisor uses live query logs; it
does not automatically ingest labelled benchmark misses.

Project Managers see only non-admin activity in their department scope. System
Admin sees the global operational view.

## Settings: System Admin

1. Open **Settings**.
2. Review **Current Architecture**.
3. Review **LLM Health**. The configuration check is token-free.
4. Use **Test** only when a small billable/live request is acceptable.
5. Expand **Usage Records** for recent Azure token and estimated-cost records.
6. Select storage, vector and LLM providers.
7. Leave embedding locked to `local`.
8. Adjust Top-K, relevance threshold or guardrail only with evaluation evidence.
9. Select **Save Runtime Settings**.
10. If a vector-provider change is pending, run the required full rebuild at a
    controlled time.

Vector/embedding changes are risky because indexed vectors must match the query
embedding model and active schema. The system keeps the previous active setting
until the required rebuild succeeds.

## Sign Out and End a Test

1. Wait for active chat/index/connector jobs to finish.
2. Select **Logout**.
3. Close the browser on shared machines.
4. The operator stops Streamlit/FastAPI only after all testers finish.
5. Follow the UAT retention plan for chat/query logs and uploaded documents.
