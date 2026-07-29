# Azure Setup Guide

This guide recreates the current cost-conscious hybrid profile with
company-owned Azure resources:

```text
Azure Blob Storage
  -> local extraction and local MiniLM embeddings
  -> Azure AI Search Free
  -> Azure OpenAI gpt-5.4-nano deployment
  -> FastAPI
  -> Streamlit and Teams clients
```

SQLite remains local and stores governance metadata, ACLs, versions, jobs, chat
history and logs. Azure AI Search is not the metadata system of record.

Portal names can change. Use the linked Microsoft documentation and verify the
resource Summary/Review page before creating anything billable.

## Security First: Rotate an Exposed Storage Key

If an Azure Storage connection string or `AccountKey` has been pasted into chat,
email, source control or a screenshot, treat it as compromised.

1. Open Azure Portal.
2. Open the affected **Storage account**.
3. Open **Security + networking → Access keys**.
4. Determine whether the application currently uses `key1` or `key2`.
5. Regenerate the unused key first.
6. Update `.env` to the regenerated key's connection string and verify upload.
7. Regenerate the previously exposed key.
8. Verify upload again and review Storage activity/costs.

Do not place the replacement key in documentation or source control.

## 1. Create a Resource Group

1. Sign in to [Azure Portal](https://portal.azure.com/).
2. Search for **Resource groups**.
3. Select **Create**.
4. Choose the company subscription.
5. Use a clear name such as `rg-rag-copilot-uat`.
6. Select a region allowed by company data policy and supported by the required
   services.
7. Review and create.

Using one resource group makes cost review and later cleanup easier.

## 2. Create Azure Blob Storage

### Create the storage account

1. Search for **Storage accounts** and select **Create**.
2. Select the UAT subscription and resource group.
3. Choose a globally unique storage account name.
4. Select **Standard** performance.
5. Select **Locally-redundant storage (LRS)** for a small non-production UAT
   unless company policy requires stronger redundancy.
6. Keep secure transfer required and public blob access disabled.
7. Review networking and data-protection settings with company IT.
8. Create the account.

General-purpose v2 is the standard storage-account type for Blob Storage. See
[Storage account overview](https://learn.microsoft.com/azure/storage/common/storage-account-overview).

### Create the private container

1. Open the storage account.
2. Open **Data storage → Containers**.
3. Select **+ Container**.
4. Name it `kb-documents` or another lowercase name.
5. Set anonymous access to **Private (no anonymous access)**.
6. Select **Create**.

Reference: [Manage blob containers in Azure Portal](https://learn.microsoft.com/azure/storage/blobs/blob-containers-portal).

### Obtain the current implementation credential

1. Open **Security + networking → Access keys**.
2. Reveal one connection string only in the approved secure environment.
3. Put it in local `.env`:

```dotenv
STORAGE_BACKEND=azure_blob
AZURE_STORAGE_CONNECTION_STRING=<SECRET_CONNECTION_STRING>
AZURE_STORAGE_CONTAINER_NAME=kb-documents
```

Microsoft documents the portal location and format in
[Configure a storage connection string](https://learn.microsoft.com/azure/storage/common/storage-configure-connection-string).

The current code requires this connection string. Managed identity is a future
hardening path.

## 3. Create Azure AI Search Free

1. In Azure Portal, search for **Azure AI Search**.
2. Select **Create**.
3. Choose the same UAT subscription and resource group.
4. Choose a globally unique service name.
5. Select an approved region.
6. Select the **Free** tier. Do not select Standard for this controlled UAT.
7. Review and create.

The service endpoint follows this format:

```text
https://<SEARCH_SERVICE_NAME>.search.windows.net
```

Reference: [Create an Azure AI Search service](https://learn.microsoft.com/azure/search/search-create-service-portal).

### Obtain endpoint and admin key

1. Open the Search service **Overview** page and copy the URL.
2. Open **Settings → Keys**.
3. Copy one admin key through the approved secret channel.
4. Configure:

```dotenv
VECTOR_BACKEND=azure_search
AZURE_SEARCH_ENDPOINT=https://<SEARCH_SERVICE_NAME>.search.windows.net
AZURE_SEARCH_ADMIN_KEY=<SECRET_ADMIN_KEY>
AZURE_SEARCH_INDEX_NAME=rag-copilot-documents
```

The current backend needs an admin key because it manages index schema and
documents. Azure recommends Entra ID/RBAC for stronger production security, but
the current application has not implemented that authentication path. See
[Search service configuration](https://learn.microsoft.com/azure/search/search-manage).

### Free-tier constraints

- use one small UAT index;
- keep the total service storage below 50 MB;
- do not depend on dedicated throughput or an SLA;
- monitor index size and query throttling;
- do not silently upgrade to a paid tier.

The Free tier uses shared infrastructure and cannot scale out. It is suitable
for this limited synthetic-data evaluation, not enterprise-scale production.

## 4. Create a Microsoft Foundry Resource and Project

1. Open [Microsoft Foundry](https://ai.azure.com/).
2. Select or create a Foundry resource in the UAT resource group.
3. Create/select a project connected to that resource if the portal requests
   one.
4. Confirm the subscription, resource group, region and billing context.

The project is the working portal context. The deployed model and endpoint are
backed by the Foundry resource.

## 5. Deploy `gpt-5.4-nano`

1. In Foundry, open **Model catalog** or **Models + endpoints**.
2. Search for `gpt-5.4-nano`.
3. Open the model card and confirm it is an Azure-sold OpenAI model available to
   the company subscription and region.
4. Select **Deploy**.
5. Choose a Standard/pay-as-you-go deployment rather than provisioned throughput.
6. Use a stable deployment name such as `rag-gpt-5.4-nano`.
7. Keep capacity/quota conservative for UAT.
8. Review content-filter and data-processing selections.
9. Select **Deploy** and wait for the deployment state to become successful.
10. Open the deployment playground and send one short test prompt.

Foundry deployments have a model, model version, deployment name, capacity type
and rate limit. The application uses the deployment name in the API request. See
[Foundry model endpoints and deployments](https://learn.microsoft.com/azure/foundry/foundry-models/concepts/endpoints).

### Obtain endpoint, key and deployment name

From the deployment/resource details, record through an approved secret channel:

- OpenAI-compatible endpoint;
- API key;
- deployment name.

Configure:

```dotenv
LLM_BACKEND=azure_openai
AZURE_OPENAI_ENDPOINT=https://<FOUNDRY_RESOURCE>.services.ai.azure.com/openai/v1
AZURE_OPENAI_API_KEY=<SECRET_API_KEY>
AZURE_OPENAI_CHAT_DEPLOYMENT=rag-gpt-5.4-nano
```

The endpoint must be the OpenAI v1 base expected by this application, not only a
project-management endpoint and not the older `/models` inference route.

### Token-spending connectivity test

After protecting `.env` and activating the virtual environment:

```bash
python - <<'PY'
from src.rag.llm_factory import create_chat_llm

llm = create_chat_llm()
print(type(llm).__name__)
print(llm.invoke("Reply with exactly: azure ok"))
PY
```

Expected class: `FoundryOpenAIModel`. This sends a real request and consumes a
small number of tokens. A class-name-only check is token-free.

## 6. Activate the Hybrid Profile

Complete `.env`:

```dotenv
STORAGE_BACKEND=azure_blob
VECTOR_BACKEND=azure_search
EMBEDDING_BACKEND=local
LLM_BACKEND=azure_openai
```

Restart FastAPI and Streamlit. Then sign in as System Admin:

1. Open **Settings**.
2. Confirm Storage=`azure_blob`, Vector=`azure_search`, Embedding=`local`,
   LLM=`azure_openai`.
3. Save the intended runtime settings if SQLite still contains older values.
4. Run the LLM config check.
5. Run a full Search rebuild only when intentionally creating/switching the
   target index.
6. Ask a known query and inspect citations.

## 7. Cost Controls

1. Open Azure **Cost Management + Billing**.
2. Create a resource-group budget for the UAT group.
3. Add notifications at conservative thresholds such as USD 5, 10 and 20.
4. Review Cost Analysis daily during company testing.
5. Monitor Foundry input/output tokens and HTTP 429 responses.
6. Monitor Blob capacity/operations and Search storage.
7. Stop the UAT when the agreed testing window ends.

Azure budget alerts notify; they do not automatically stop resources. If a hard
stop is required, company administrators must design approved automation.

## 8. Azure Acceptance Checklist

- [ ] Storage container is private.
- [ ] No secret appears in Git, documentation or screenshots.
- [ ] One test upload creates a Blob object and SQLite storage URI.
- [ ] Search Settings shows the Free tier.
- [ ] Search index is created and contains expected chunks.
- [ ] Embedding backend remains `local`.
- [ ] Foundry deployment responds to one short test.
- [ ] A portal chat returns a grounded answer with expected sources.
- [ ] A restricted query remains blocked before the LLM.
- [ ] Budget alerts and responsible contacts are configured.
