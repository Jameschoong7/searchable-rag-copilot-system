# Microsoft Graph Connector Setup

The current connector is delegated and read-only:

```text
Administrator runs device-code login
  -> MSAL caches delegated tokens locally
  -> FastAPI reads signed-in user's OneDrive and OneNote
  -> administrator scans and stages selected items
  -> metadata/ACL review
  -> approval
  -> index update
```

Graph is used only for ingestion. It is not the portal's user-authentication
system. SharePoint live integration is not implemented.

## Required Permissions

Use delegated Microsoft Graph permissions:

- `User.Read`
- `Files.Read`
- `Notes.Read`

These permissions allow the signed-in user to read their profile, OneDrive
files and OneNote content. The application does not require write access.

Microsoft documents that `Files.Read` permits reading the signed-in user's files
and that `Notes.Read` is the least-privileged OneNote read scope:

- [OneDrive permission scopes](https://learn.microsoft.com/onedrive/developer/rest-api/concepts/permissions_reference?view=odsp-graph-online)
- [Get OneNote content](https://learn.microsoft.com/graph/onenote-get-content)

## Optional: Validate Access With Graph Explorer

[Microsoft Graph Explorer](https://developer.microsoft.com/graph/graph-explorer)
is useful for proving that the intended Microsoft account can read OneDrive and
OneNote before debugging the Python connector.

1. Open Graph Explorer and sign in with the same Microsoft account that will be
   used for connector testing.
2. Select the profile avatar → **Consent to permissions**, or run a query and
   open **Modify permissions**.
3. Consent only to:
   - `User.Read`;
   - `Files.Read`;
   - `Notes.Read`.
4. Run:

```http
GET https://graph.microsoft.com/v1.0/me
```

5. Verify the returned display name/account.
6. Test the restricted OneDrive root:

```http
GET https://graph.microsoft.com/v1.0/me/drive/root:/Enterprise Knowledge Base:/children
```

7. Test OneNote notebooks:

```http
GET https://graph.microsoft.com/v1.0/me/onenote/notebooks
```

Successful requests should return HTTP 200 and data belonging to the signed-in
account. Microsoft documents Graph Explorer's permission-consent controls in
[Work with Graph Explorer](https://learn.microsoft.com/graph/graph-explorer/graph-explorer-features).

### Graph Explorer consent is separate

Graph Explorer and Searchable RAG Copilot are two different Entra applications:

```text
Graph Explorer consent
  -> authorizes Microsoft's Graph Explorer application

Entra app registration + device login
  -> authorizes this project's GRAPH_CLIENT_ID
```

Therefore, successful Graph Explorer queries prove that the account, content and
permission type can work, but they do not authorize the Python connector. The
connector still needs its own app registration permissions and device-code
consent described below.

Do not copy Graph Explorer's **Access token** into `.env`. That token belongs to
Graph Explorer's client, can expire, and may use an opaque format. The project
should obtain its own token through MSAL and `scripts/graph_device_login.py`.

## 1. Create the Entra App Registration

1. Sign in to [Microsoft Entra admin center](https://entra.microsoft.com/).
2. Switch to the tenant/account that owns the intended OneDrive/OneNote data.
3. Open **Identity → Applications → App registrations**.
4. Select **New registration**.
5. Name it, for example, `Searchable RAG Copilot Connector`.
6. Choose the supported account type:
   - current personal setup: personal Microsoft accounts or the broad
     organizational-plus-personal option;
   - company setup: accounts in the company directory only.
7. Leave Redirect URI empty for this device-code client.
8. Select **Register**.
9. Copy the **Application (client) ID**. Do not use the Object ID.

Reference: [Register an application](https://learn.microsoft.com/graph/auth-register-app-v2).

## 2. Enable Public Client Flow

Device-code login requires a public client application.

1. Open the app registration.
2. Open **Authentication**.
3. Find **Advanced settings**.
4. Set **Allow public client flows** to **Yes**.
5. Save.

The device-code flow is available only for public clients. See
[MSAL authentication flows](https://learn.microsoft.com/entra/msal/msal-authentication-flows).

Do not create a client secret for the current connector. A public client cannot
safely protect a secret installed on the user's machine.

## 3. Add Delegated Graph Permissions

1. Open **API permissions**.
2. Select **Add a permission**.
3. Select **Microsoft Graph**.
4. Select **Delegated permissions**.
5. Add `User.Read`, `Files.Read` and `Notes.Read`.
6. Save the permissions.
7. If tenant policy requires administrator consent, ask the company tenant
   administrator to review and grant only these delegated scopes.

Do not add `Files.ReadWrite`, `Notes.ReadWrite` or broad application permissions
for this UAT.

## 4. Configure `.env`

For the current personal Microsoft account profile:

```dotenv
GRAPH_CONNECTOR_ENABLED=true
GRAPH_CLIENT_ID=<APPLICATION_CLIENT_ID>
GRAPH_AUTHORITY=https://login.microsoftonline.com/consumers
GRAPH_SCOPES=User.Read Files.Read Notes.Read
GRAPH_ONEDRIVE_ROOT_PATH=/Enterprise Knowledge Base
GRAPH_ONENOTE_NOTEBOOK_FILTER=Enterprise Knowledge Base
```

For a company single tenant:

```dotenv
GRAPH_AUTHORITY=https://login.microsoftonline.com/<COMPANY_TENANT_ID>
```

Do not set `GRAPH_ACCESS_TOKEN` during normal operation. A manually pasted token
can be opaque or JWT-shaped, expires quickly and prevents the MSAL cache from
performing the intended silent refresh path.

## 5. Prepare OneDrive Structure

Create this folder hierarchy in the signed-in user's OneDrive:

```text
Enterprise Knowledge Base/
├── ENGINEERING/
├── HR/
├── IT/
├── OPERATIONS/
└── SECURITY/
```

Place supported TXT, PDF and DOCX files under the department folders. The
connector starts at `GRAPH_ONEDRIVE_ROOT_PATH`; it does not need to list the
user's whole OneDrive.

Department inference is case-insensitive and normalizes folder labels to the
system's supported department values. Files outside a recognized department
folder require careful metadata review before approval.

## 6. Prepare OneNote Structure

Create a notebook named exactly:

```text
Enterprise Knowledge Base
```

Create sections that identify the department, for example:

```text
Enterprise Knowledge Base
├── ENGINEERING
├── HR
├── IT
├── OPERATIONS
└── SECURITY
```

Add knowledge pages beneath the appropriate section. The notebook filter limits
discovery to this notebook; section/notebook/path information is used for
department inference and metadata review.

The Graph OneNote API returns page HTML. The connector normalizes page text into
the same ingestion pipeline used by other sources.

## 7. Run Device Login

From the repository root:

```bash
source .venv/bin/activate
python scripts/graph_device_login.py
```

The terminal prints a Microsoft device-login URL and one-time code:

1. Open the URL in a browser.
2. Enter the code.
3. Sign in with the OneDrive/OneNote owner account.
4. Review the requested delegated permissions.
5. Return to the terminal and wait for success.

MSAL saves a sensitive token cache under:

```text
data/auth/graph_token_cache.bin
```

The cache is gitignored and must not be distributed. Later requests call
`acquire_token_silent()` and refresh when Microsoft permits it. Run device login
again only when the cache is missing, revoked or can no longer refresh.

## 8. Verify Graph Before Opening Streamlit

```bash
source .venv/bin/activate
python - <<'PY'
from src.connectors.graph_client import get_current_graph_user
print(get_current_graph_user())
PY
```

Then verify the configured OneDrive root:

```bash
python - <<'PY'
from src.connectors.graph_client import list_onedrive_root_children

for item in list_onedrive_root_children():
    print("folder" if "folder" in item else "file", item.get("name"))
PY
```

Do not paste returned access tokens or personal identifiers into support tickets.

## 9. Ingest OneDrive or OneNote Through the Portal

1. Start FastAPI after device login so it reads the same `.env` and token cache.
2. Sign in as System Admin.
3. Open **KB Management → OneDrive** or **OneNote**.
4. Select **Scan**.
5. Expand discovered items only after a successful scan.
6. Select new/rejected items and start the batch stage job.
7. Open **Review Queue**.
8. Confirm title, department, category, tags, allowed roles and departments.
9. Approve the item.
10. Open **Index Sync** and update pending documents.
11. Ask a known question and verify the connector source appears in citations.

Staging downloads/normalizes the source into the application's storage and
creates an inactive `pending_review` metadata record. It is not searchable until
approval and indexing.

## 10. Refresh Changed Content

1. Edit the same OneDrive item or OneNote page.
2. Scan the connector again to obtain current upstream metadata.
3. Select the indexed source in the refresh list.
4. Run refresh.
5. If the content hash changed, the backend creates a new pending version and
   archives the previous active version.
6. Run the pending index update.
7. Verify the new answer and confirm old content is not retrieved.

Replacing a PDF in OneDrive can preserve or change its Graph item identity
depending on how the replacement is performed. Replace the existing file in
place where possible and verify the discovered item before refreshing.

## Troubleshooting Authentication

| Symptom | Likely cause | Action |
|---|---|---|
| `401 Unauthorized` | Expired/manual token or wrong audience | Remove `GRAPH_ACCESS_TOKEN`, run device login, restart FastAPI. |
| `No cached Microsoft account` | Token cache missing | Run `scripts/graph_device_login.py`. |
| `invalid_client` requiring a secret | Public client flow disabled or wrong app type | Enable **Allow public client flows**; do not add a secret as a workaround. |
| OneDrive says tenant lacks SPO license | Signed into an Entra tenant without OneDrive/SharePoint license | Use the intended licensed company account or current personal consumer authority. |
| Graph Explorer works but app fails | Explorer and app tokens represent different accounts/tenants/apps | Compare `/me`, authority, client ID and scopes. |
| OneNote empty | Notebook filter mismatch | Match the notebook name exactly or temporarily verify without a filter. |
| Portal connector gets 401 while Python test works | FastAPI started before token/config refresh | Restart FastAPI. |

## Privacy and Handover

- use a dedicated UAT account where possible;
- scope OneDrive scanning to `/Enterprise Knowledge Base`;
- scope OneNote scanning to one notebook;
- do not distribute `data/auth/graph_token_cache.bin`;
- do not include personal Graph item IDs in the clean sample package;
- remove/revoke the app's consent and delete the local token cache after the UAT
  if the company no longer needs the connector.
