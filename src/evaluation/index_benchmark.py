import json
import os
from pathlib import Path
import sys
import time

from src.core.config import AppConfig, read_app_config
from src.vector.factory import get_vector_backend, get_vector_backend_for_config
from dotenv import load_dotenv

from src.metadata.repository import load_document_metadata


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATED_DATA_PATH = PROJECT_ROOT / "data/simulated"
BENCHMARK_RESULTS_PATH = PROJECT_ROOT / "data/evaluation/index_benchmark_results.json"
BENCHMARK_HISTORY_PATH = PROJECT_ROOT / "data/evaluation/index_benchmark_history.json"

def calculate_optional_delta(after_value, before_value):
    """Return a numeric delta only when both values are measurable."""
    if after_value is None or before_value is None:
        return None

    return after_value - before_value


def count_simulated_source_files(path: Path) -> int:
    """Count searchable local source files in the simulated document folder."""
    supported_suffixes = {".txt", ".pdf", ".docx"}

    return sum(
        1
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in supported_suffixes
    )


def calculate_index_delta(after_snapshot: dict, before_snapshot: dict) -> dict:
    """Calculate backend-neutral index count and size changes."""
    return {
        "indexed_chunk_count": (
            after_snapshot["indexed_chunk_count"]
            - before_snapshot["indexed_chunk_count"]
        ),
        "index_size_bytes": calculate_optional_delta(
            after_snapshot["index_size_bytes"],
            before_snapshot["index_size_bytes"],
        ),
        "index_size_mb": calculate_optional_delta(
            after_snapshot["index_size_mb"],
            before_snapshot["index_size_mb"],
        ),
    }


def build_index_benchmark_snapshot(config: AppConfig | None = None) -> dict:
    """Create one benchmark snapshot for metadata, source files, and the selected index."""
    active_documents = load_document_metadata()
    config = config or read_app_config()
    vector_backend = get_vector_backend_for_config(config)

    index_record_count = vector_backend.get_index_record_count()
    index_size_bytes = vector_backend.get_index_size_bytes()
    index_size_mb = (
        round(index_size_bytes / (1024 * 1024), 2)
        if index_size_bytes is not None
        else None
    )

    return {
        "vector_backend": config.vector_backend,
        "active_metadata_records": len(active_documents),
        "simulated_source_files": count_simulated_source_files(SIMULATED_DATA_PATH),
        "indexed_chunk_count": index_record_count,
        "index_size_bytes": index_size_bytes,
        "index_size_mb": index_size_mb,
        "index_size_available": index_size_bytes is not None,

        # Backward-compatible Chroma fields for older dashboard/history rows.
        "chroma_vector_count": index_record_count if config.vector_backend == "chroma" else None,
        "chroma_db_size_bytes": index_size_bytes if config.vector_backend == "chroma" else None,
        "chroma_db_size_mb": index_size_mb if config.vector_backend == "chroma" else None,
    }


def build_full_rebuild_benchmark(config: AppConfig | None = None) -> dict:
    """Measure the cost and storage result of rebuilding the selected vector index."""
    from src.etl.pipeline import rebuild_vector_store

    config = config or read_app_config()
    vector_backend = get_vector_backend_for_config(config)

    before_snapshot = build_index_benchmark_snapshot(config)

    start_time = time.perf_counter()
    rebuild_result = rebuild_vector_store(vector_backend=vector_backend)
    elapsed_seconds = round(time.perf_counter() - start_time, 3)

    after_snapshot = build_index_benchmark_snapshot(config)

    return {
        "benchmark_type": "full_rebuild",
        "elapsed_seconds": elapsed_seconds,
        "before": before_snapshot,
        "rebuild_result": rebuild_result,
        "after": after_snapshot,
        "delta": calculate_index_delta(after_snapshot, before_snapshot),
    }


def build_single_document_update_benchmark(source_path: str) -> dict:
    """Measure the cost of deleting and re-indexing one source document only."""
    from src.etl.pipeline import delete_vectors_for_source, index_single_document

    before_snapshot = build_index_benchmark_snapshot()

    start_time = time.perf_counter()

    deleted_vector_count = delete_vectors_for_source(source_path)
    index_result = index_single_document(source_path)

    elapsed_seconds = round(time.perf_counter() - start_time, 3)

    after_snapshot = build_index_benchmark_snapshot()

    return {
        "benchmark_type": "single_document_update",
        "source": source_path,
        "elapsed_seconds": elapsed_seconds,
        "before": before_snapshot,
        "deleted_vector_count": deleted_vector_count,
        "index_result": index_result,
        "after": after_snapshot,
        "delta": calculate_index_delta(after_snapshot, before_snapshot),
    }


def build_batch_update_benchmark(source_paths: list[str]) -> dict:
    """Measure the cost of updating multiple changed source documents."""
    from src.etl.pipeline import index_changed_documents

    before_snapshot = build_index_benchmark_snapshot()

    start_time = time.perf_counter()
    update_result = index_changed_documents(source_paths)
    elapsed_seconds = round(time.perf_counter() - start_time, 3)

    after_snapshot = build_index_benchmark_snapshot()

    before_active_vectors = before_snapshot["indexed_chunk_count"]

    estimated_unchanged_chunks_avoided = max(
        before_active_vectors - update_result["total_chunks_indexed"],
        0,
    )

    return {
        "benchmark_type": "batch_incremental_update",
        "changed_document_count": update_result["changed_document_count"],
        "updated_sources": update_result["updated_sources"],
        "elapsed_seconds": elapsed_seconds,
        "before": before_snapshot,
        "update_results": update_result["update_results"],
        "total_deleted_vectors": update_result["total_deleted_vectors"],
        "total_document_objects_loaded": update_result["total_document_objects_loaded"],
        "total_chunks_indexed": update_result["total_chunks_indexed"],
        "estimated_unchanged_chunks_avoided": estimated_unchanged_chunks_avoided,
        "after": after_snapshot,
        "delta": calculate_index_delta(after_snapshot, before_snapshot),
    }


def save_benchmark_result(result: dict) -> None:
    """Save the latest index benchmark result for dashboard/report evidence."""
    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BENCHMARK_RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(result, results_file, indent=2)
    
    save_benchmark_history_entry(result)


def load_benchmark_history() -> list[dict]:
    """Load saved benchmark history for comparing rebuild and update runs."""
    if not BENCHMARK_HISTORY_PATH.exists():
        return []

    with BENCHMARK_HISTORY_PATH.open("r", encoding="utf-8") as history_file:
        return json.load(history_file)


def save_benchmark_history_entry(result: dict) -> None:
    """Append one benchmark result so dashboard comparisons are not hardcoded."""
    BENCHMARK_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = load_benchmark_history()
    history.append(result)

    with BENCHMARK_HISTORY_PATH.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)


if __name__ == "__main__":
    if "--full-rebuild" in sys.argv:
        snapshot = build_full_rebuild_benchmark()
    elif "--single-document" in sys.argv:
        source_argument_index = sys.argv.index("--single-document") + 1

        if source_argument_index >= len(sys.argv):
            raise SystemExit("Usage: python -m src.evaluation.index_benchmark --single-document <source_path>")

        snapshot = build_single_document_update_benchmark(
            sys.argv[source_argument_index]
        )
    elif "--batch-update" in sys.argv:
        source_argument_index = sys.argv.index("--batch-update") + 1
        source_paths = sys.argv[source_argument_index:]

        if not source_paths:
            raise SystemExit(
                "Usage: python -m src.evaluation.index_benchmark "
                "--batch-update <source_path> [<source_path> ...]"
            )

        snapshot = build_batch_update_benchmark(source_paths)
    else:
        snapshot = build_index_benchmark_snapshot()

    save_benchmark_result(snapshot)

    print(json.dumps(snapshot, indent=2))
    print(f"Saved benchmark result to {BENCHMARK_RESULTS_PATH}")
