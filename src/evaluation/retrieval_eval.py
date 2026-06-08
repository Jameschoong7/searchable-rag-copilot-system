import json
from pathlib import Path

from src.rag.engine import retrieve_relevant_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELLED_QUERIES_PATH = PROJECT_ROOT / "data/evaluation/labelled_queries.json"
RESULTS_PATH = PROJECT_ROOT / "data/evaluation/retrieval_eval_results.json"


def load_labelled_queries() -> list[dict]:
    """Load curated retrieval test cases with expected source filenames."""
    with LABELLED_QUERIES_PATH.open("r", encoding="utf-8") as query_file:
        return json.load(query_file)


def source_filename_from_chunk(chunk) -> str:
    """Extract only the filename from a retrieved chunk source path."""
    source_path = chunk.metadata.get("source", "")
    return Path(source_path).name


def evaluate_query(test_case: dict) -> dict:
    """Run one labelled query and check if expected source appears in top 5 chunks."""
    chunks = retrieve_relevant_chunks(
        question=test_case["question"],
        role=test_case["role"],
        department=test_case["department"],
        department_filter=test_case.get("department_filter"),
        file_type_filter=test_case.get("file_type_filter"),
        top_k=5,
    )

    retrieved_sources = list(
        dict.fromkeys(source_filename_from_chunk(chunk) for chunk in chunks)
    )

    expected_source = test_case["expected_source"]
    hit = expected_source in retrieved_sources

    return {
        "query_id": test_case["query_id"],
        "suite": test_case.get("suite", "default"),
        "question": test_case["question"],
        "expected_source": expected_source,
        "retrieved_sources": retrieved_sources,
        "hit": hit,
        "issue": "" if hit else "Expected source was not found in retrieved top-K chunks.",
    }


def summarize_results(results: list[dict]) -> dict:
    """Calculate overall and per-suite retrieval accuracy summaries."""
    total_queries = len(results)
    correct_queries = sum(1 for result in results if result["hit"])
    top_k_accuracy = correct_queries / total_queries if total_queries else 0

    by_suite = {}

    for result in results:
        suite_name = result["suite"]

        if suite_name not in by_suite:
            by_suite[suite_name] = {
                "total_queries": 0,
                "correct_queries": 0,
                "miss_count": 0,
                "top_k_accuracy": 0,
                "top_k_accuracy_percent": 0,
            }

        by_suite[suite_name]["total_queries"] += 1

        if result["hit"]:
            by_suite[suite_name]["correct_queries"] += 1
        else:
            by_suite[suite_name]["miss_count"] += 1

    for suite_summary in by_suite.values():
        suite_total = suite_summary["total_queries"]
        suite_correct = suite_summary["correct_queries"]
        suite_accuracy = suite_correct / suite_total if suite_total else 0

        suite_summary["top_k_accuracy"] = round(suite_accuracy, 4)
        suite_summary["top_k_accuracy_percent"] = round(suite_accuracy * 100, 2)

    return {
        "overall": {
            "total_queries": total_queries,
            "correct_queries": correct_queries,
            "miss_count": total_queries - correct_queries,
            "top_k_accuracy": round(top_k_accuracy, 4),
            "top_k_accuracy_percent": round(top_k_accuracy * 100, 2),
        },
        "by_suite": by_suite,
    }


def run_evaluation() -> dict:
    """Run all labelled retrieval cases and calculate Top-K Accuracy."""
    test_cases = load_labelled_queries()
    results = [evaluate_query(test_case) for test_case in test_cases]
    summary = summarize_results(results)

    miss_rows = [
        result for result in results
        if not result["hit"]
    ]

    return {
        "summary": summary,
        "miss_rows": miss_rows,
        "results": results,
    }



if __name__ == "__main__":
    evaluation_output = run_evaluation()

    with RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(evaluation_output, results_file, indent=2)

    print(json.dumps(evaluation_output["summary"], indent=2))
    print(f"Saved full results to {RESULTS_PATH}")