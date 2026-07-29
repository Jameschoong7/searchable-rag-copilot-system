# Security and Limitations

## Deployment Classification

The current system is appropriate for:

- local development;
- supervised demonstrations;
- controlled company-side UAT with named testers;
- synthetic/public/approved non-confidential documents.

It must not yet be represented as an unrestricted production deployment.

## Current Security Controls

- portal passwords are PBKDF2-hashed in SQLite;
- role and department ACL filtering occurs before authorized chunks are sent to
  the LLM;
- document metadata/ACL is reviewed before connector/ZIP content is searchable;
- old document versions are archived/excluded from active retrieval;
- `.env`, runtime databases, logs and Graph cache are gitignored;
- OneDrive/OneNote connectors request delegated read-only scopes;
- Graph scanning is restricted to configured root/notebook scope;
- Blob container guidance requires private access;
- LLM usage is recorded for administrator visibility.

## High-Priority Limitations

### API identity is not production-authenticated

FastAPI request payloads currently include user, role and department supplied by
the client. Streamlit binds those values to the demo login, but the API itself
does not validate an Entra JWT/OIDC identity on every request.

Consequence: do not expose FastAPI directly to untrusted users or the public
internet. A direct caller could assert a more privileged role.

### Demo accounts are shared/static

Seeded users use a common initial UAT password. There is no UI for password reset,
account provisioning, MFA, lockout or identity lifecycle.

### Teams profile commands are simulations

Commands such as `/use-it-manager` select a simulated test profile. They are not
Microsoft Teams identity-to-role mapping and must be disabled/replaced before
real deployment.

### No rate limiting or abuse protection

The API has no per-user rate limit, request quota, WAF or robust concurrency
control. An unrestricted deployment could consume tokens or overload SQLite/jobs.

### Background execution is not restart-durable

Job records persist in SQLite, but active FastAPI BackgroundTasks do not resume
after process failure/restart. There is no external queue/worker.

### SQLite is a single-node UAT database

The system has not been proved for multiple API workers or high write
concurrency. Do not launch multiple Uvicorn workers against the same database
without a migration/concurrency design.

### Hosting controls are absent

The repository does not include production TLS, reverse proxy, container image,
process supervisor, secret manager integration or private networking deployment.

### Logs contain sensitive operational text

Query logs and chat memory can store usernames, questions, answers and source
paths. There is no automatic retention/deletion policy.

### Upload security is limited

The application validates type/size/path constraints but does not provide
enterprise malware scanning, DLP or content moderation for uploaded files.

### Connector limitations

- Graph uses one delegated cached account;
- pagination/delta synchronization is incomplete;
- live SharePoint integration is not available;
- enterprise SharePoint ACL mirroring is not implemented;
- OneNote requires delegated authentication and licensed/accessible content.

### Citation limitations

Citations are source-path based and can combine several chunks into unique source
labels. Exact page/paragraph anchoring and clickable original preview are not
guaranteed.

## Required Controlled-UAT Safeguards

- named testers only;
- synthetic/public/approved non-confidential data;
- one controlled machine or restricted company network;
- do not expose FastAPI directly;
- company-owned cloud resources;
- private Blob container;
- conservative Azure budget alerts;
- daily usage/error review;
- no secrets in screenshots/log bundles;
- pre-UAT backup and post-UAT cleanup;
- agreed retention and responsible administrator.

## Production Hardening Roadmap

Before production:

1. Add Entra authentication to both clients and FastAPI.
2. Derive role/department from validated server-side identity/claims or a trusted
   user directory.
3. Remove demo passwords and Teams profile commands.
4. Centralize endpoint authorization policies.
5. Add TLS, reverse proxy/API gateway and private networking.
6. Move secrets to an approved secret manager and prefer managed identities.
7. Add per-user/tenant rate limiting, quotas and 429 retry/backoff.
8. Replace in-process jobs with a durable queue/worker and idempotent recovery.
9. Move SQLite to a supported managed relational database for concurrent use.
10. Add malware scanning, DLP, content-type verification and retention controls.
11. Add structured audit events and security monitoring.
12. Implement backup/restore drills and disaster recovery.
13. Add pagination/delta sync and enterprise connector ACL mapping.
14. Complete load, penetration, privacy and acceptance testing.

## Secret Response Procedure

If a key/token is exposed:

1. stop sharing/repeating it;
2. identify the resource and credential slot;
3. rotate/regenerate through the provider portal;
4. update the approved local secret store;
5. restart and verify the application;
6. rotate the second key where relevant;
7. review access/activity/cost logs;
8. search Git history and artifacts for the secret pattern;
9. document the incident without recording the secret itself.

Deleting a message or Git line is not enough after a credential has been exposed.
