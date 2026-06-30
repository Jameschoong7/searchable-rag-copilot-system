# Pending Index Update Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "Run Update for Pending Documents" run as a durable backend job so Streamlit navigation/reruns do not interrupt incremental indexing.

**Architecture:** Reuse the existing SQLite-backed `app_jobs` system used by chat jobs and full rebuild jobs. Move the synchronous pending-index implementation into a helper, expose both the existing synchronous endpoint and a new job endpoint, then have Streamlit submit/poll the job instead of blocking on the button click.

**Tech Stack:** FastAPI `BackgroundTasks`, SQLite job repository, Streamlit fragments, existing metadata/vector indexing helpers.

---

### Task 1: Add Pending Index Job Type

**Files:**
- Modify: `src/core/job_repository.py`

- [ ] **Step 1: Add a job type constant**

Add this beside the existing `JOB_TYPE_REINDEX` and `JOB_TYPE_CHAT_QUERY` constants:

```python
JOB_TYPE_INDEX_UPDATE = "index_update"
```

- [ ] **Step 2: Syntax check**

Run:

```bash
python -m py_compile src/core/job_repository.py
```

Expected: no output and exit code `0`.

---

### Task 2: Refactor Pending Index Work Into A Helper

**Files:**
- Modify: `src/api/main.py`

- [ ] **Step 1: Import the new job type**

In the `src.core.job_repository` import block, add:

```python
JOB_TYPE_INDEX_UPDATE,
```

- [ ] **Step 2: Add a job request model**

Add this after `IndexUpdatesRequest`:

```python
class IndexUpdatesJobRequest(IndexUpdatesRequest):
    """Represent a durable pending-index update request submitted as a backend job."""

    user: str
```

- [ ] **Step 3: Create the shared helper**

Move the body of the current `index_pending_document_updates()` route into a new helper above the route:

```python
def run_pending_index_update(updated_by: str) -> dict:
    """Index active pending documents and persist benchmark/update metadata."""
    import time
    from pathlib import Path

    from src.etl.pipeline import index_changed_documents_with_cleanup
    from src.evaluation.index_benchmark import (
        build_index_benchmark_snapshot,
        calculate_index_delta,
        save_benchmark_result,
    )
    from src.metadata.repository import (
        load_pending_index_documents,
        load_replaced_documents_for_new_versions,
        mark_documents_indexed,
    )

    pending_documents = load_pending_index_documents()

    if not pending_documents:
        return {
            "status": "no_pending_documents",
            "pending_document_count": 0,
            "updated_sources": [],
            "total_deleted_vectors": 0,
            "total_chunks_indexed": 0,
            "elapsed_seconds": 0,
            "message": "No pending document updates require indexing.",
        }

    pending_document_ids = [
        document["document_id"]
        for document in pending_documents
    ]

    replaced_documents = load_replaced_documents_for_new_versions(
        pending_document_ids
    )

    index_source_paths = [
        str(Path("data/simulated") / document["filename"])
        for document in pending_documents
    ]

    replaced_source_paths = [
        str(Path("data/simulated") / document["filename"])
        for document in replaced_documents
    ]

    cleanup_source_paths = replaced_source_paths + index_source_paths

    before_snapshot = build_index_benchmark_snapshot()

    start_time = time.perf_counter()
    update_result = index_changed_documents_with_cleanup(
        index_source_paths=index_source_paths,
        cleanup_source_paths=cleanup_source_paths,
    )
    elapsed_seconds = round(time.perf_counter() - start_time, 3)

    after_snapshot = build_index_benchmark_snapshot()

    benchmark_result = {
        "benchmark_type": "batch_incremental_update",
        "changed_document_count": update_result["changed_document_count"],
        "updated_sources": update_result["updated_sources"],
        "cleanup_sources": update_result["cleanup_sources"],
        "elapsed_seconds": elapsed_seconds,
        "before": before_snapshot,
        "update_results": update_result["update_results"],
        "cleanup_results": update_result["cleanup_results"],
        "total_deleted_vectors": update_result["total_deleted_vectors"],
        "total_document_objects_loaded": update_result["total_document_objects_loaded"],
        "total_chunks_indexed": update_result["total_chunks_indexed"],
        "estimated_unchanged_chunks_avoided": max(
            before_snapshot["indexed_chunk_count"] - update_result["total_chunks_indexed"],
            0,
        ),
        "after": after_snapshot,
        "delta": calculate_index_delta(after_snapshot, before_snapshot),
        "updated_by": updated_by,
    }

    save_benchmark_result(benchmark_result)
    mark_documents_indexed(pending_document_ids)

    return {
        "status": "success",
        "pending_document_count": len(pending_documents),
        "updated_sources": update_result["updated_sources"],
        "total_deleted_vectors": update_result["total_deleted_vectors"],
        "total_chunks_indexed": update_result["total_chunks_indexed"],
        "elapsed_seconds": elapsed_seconds,
        "message": (
            f"Indexed {len(pending_documents)} pending document(s), refreshed "
            f"{update_result['total_chunks_indexed']} chunk(s), and replaced "
            f"{update_result['total_deleted_vectors']} old vector(s)."
        ),
    }
```

- [ ] **Step 4: Simplify the existing synchronous endpoint**

Replace the current long route body with:

```python
@app.post("/admin/index-updates", response_model=IndexUpdatesResponse)
def index_pending_document_updates(request: IndexUpdatesRequest) -> IndexUpdatesResponse:
    """Run incremental indexing for active documents marked as pending index."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can index pending document updates.",
        )

    try:
        result = run_pending_index_update(updated_by=request.role)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Pending index update failed: {error}",
        ) from error

    return IndexUpdatesResponse(**result)
```

- [ ] **Step 5: Syntax check**

Run:

```bash
python -m py_compile src/api/main.py
```

Expected: no output and exit code `0`.

---

### Task 3: Add Background Job Endpoint

**Files:**
- Modify: `src/api/main.py`

- [ ] **Step 1: Add the background runner**

Add this near `run_reindex_job()`:

```python
def run_index_update_job(job_id: str, request: IndexUpdatesJobRequest) -> None:
    """Run pending document indexing in the background and store job status."""
    update_job(
        job_id,
        JOB_STATUS_RUNNING,
        "Updating pending documents in the active search index.",
    )

    try:
        result = run_pending_index_update(updated_by=request.user)

        update_job(
            job_id,
            JOB_STATUS_SUCCEEDED,
            result["message"],
            result,
        )
    except Exception as error:
        update_job(
            job_id,
            JOB_STATUS_FAILED,
            f"Pending index update failed: {error}",
            {
                "status": "failed",
                "pending_document_count": 0,
                "updated_sources": [],
                "total_deleted_vectors": 0,
                "total_chunks_indexed": 0,
                "elapsed_seconds": 0,
                "message": str(error),
            },
        )
```

- [ ] **Step 2: Add the job endpoint**

Add this near `/admin/reindex-jobs`:

```python
@app.post("/admin/index-update-jobs", response_model=JobResponse)
def create_index_update_job(
    request: IndexUpdatesJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a durable pending-index update job so Streamlit reruns do not interrupt indexing."""
    if request.role != SYSTEM_ADMIN_ROLE:
        raise HTTPException(
            status_code=403,
            detail="Only System Admin can index pending document updates.",
        )

    job = create_job(
        job_type=JOB_TYPE_INDEX_UPDATE,
        created_by=request.user,
        message="Pending document index update queued.",
    )

    background_tasks.add_task(run_index_update_job, job["job_id"], request)

    return JobResponse(**job)
```

- [ ] **Step 3: Syntax check**

Run:

```bash
python -m py_compile src/api/main.py
```

Expected: no output and exit code `0`.

---

### Task 4: Update Streamlit To Submit And Poll Pending Index Jobs

**Files:**
- Modify: `src/ui/app.py`

- [ ] **Step 1: Add endpoint constant**

Add below `REINDEX_JOBS_URL`:

```python
INDEX_UPDATE_JOBS_URL = f"{API_BASE_URL}/admin/index-update-jobs"
```

- [ ] **Step 2: Add submit helper**

Add below `submit_reindex_job()`:

```python
def submit_index_update_job() -> dict:
    """Submit pending-document indexing as a durable backend job."""
    response = requests.post(
        INDEX_UPDATE_JOBS_URL,
        json={
            "role": st.session_state["role"],
            "user": st.session_state["user"],
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()
```

- [ ] **Step 3: Add poller**

Add below `poll_active_reindex_job()`:

```python
@st.fragment(run_every="2s")
def poll_active_index_update_job() -> None:
    """Poll the active pending-index update job without blocking Streamlit navigation."""
    active_index_update_job_id = st.session_state.get("active_index_update_job_id")

    if not active_index_update_job_id:
        return

    try:
        job = get_backend_job(active_index_update_job_id)
    except requests.exceptions.RequestException as error:
        st.warning(f"Index update job status unavailable: {error}")
        return

    if job["status"] in ["queued", "running"]:
        st.info(job["message"])
        return

    if job["status"] == "succeeded":
        result = job["result"]
        st.session_state["index_update_job_message"] = result["message"]

    elif job["status"] == "failed":
        st.session_state["index_update_job_message"] = job["message"]

    st.session_state.pop("active_index_update_job_id", None)
    st.rerun()
```

- [ ] **Step 4: Call the poller**

Add below `poll_active_reindex_job()`:

```python
poll_active_index_update_job()
```

- [ ] **Step 5: Replace the synchronous KB button action**

Replace the body of the `Run Update for Pending Documents` button with:

```python
if st.button(
    "Run Update for Pending Documents",
    use_container_width=True,
    disabled=bool(st.session_state.get("active_index_update_job_id")),
):
    try:
        job = submit_index_update_job()
    except requests.exceptions.RequestException as error:
        st.error(f"Could not submit index update job: {error}")
    else:
        st.session_state["active_index_update_job_id"] = job["job_id"]
        st.session_state["index_update_job_message"] = "Pending document index update queued."
        st.rerun()
```

- [ ] **Step 6: Show the latest job message in KB Management**

Near the existing index sync controls, add:

```python
if st.session_state.get("index_update_job_message"):
    st.info(st.session_state["index_update_job_message"])
```

- [ ] **Step 7: Syntax check**

Run:

```bash
python -m py_compile src/ui/app.py
```

Expected: no output and exit code `0`.

---

### Task 5: Manual Verification

**Files:**
- Verify: FastAPI and Streamlit running locally

- [ ] **Step 1: Restart FastAPI**

Stop and restart the backend so the new endpoint is loaded.

- [ ] **Step 2: Start Streamlit**

Open the Admin Web Portal.

- [ ] **Step 3: Create a pending document**

Upload a small TXT/PDF/DOCX file as System Admin and save metadata.

- [ ] **Step 4: Start pending update**

Go to KB Management and click `Run Update for Pending Documents`.

Expected:
- button queues a job quickly
- status message appears
- switching to Performance or Chat does not cancel the job
- returning to KB Management shows completion
- pending document status changes to indexed

- [ ] **Step 5: Commit**

```bash
git add src/core/job_repository.py src/api/main.py src/ui/app.py
git commit -m "Run pending index updates as backend jobs"
```

---

## Self-Review

- Spec coverage: covers backend durable job, existing synchronous endpoint preservation, Streamlit submit/poll, and manual verification.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps.
- Type consistency: uses existing `JobResponse`, `create_job`, `update_job`, and `get_backend_job` patterns; new request model extends `IndexUpdatesRequest` consistently with `ReindexJobRequest`.
