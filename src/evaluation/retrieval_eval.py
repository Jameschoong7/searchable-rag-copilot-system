import json
from pathlib import Path

from src.rag.engine import generate_answer, retrieve_relevant_chunks_with_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELLED_QUERIES_PATH = PROJECT_ROOT / "data/evaluation/labelled_queries.json"
RESULTS_PATH = PROJECT_ROOT / "data/evaluation/retrieval_eval_results.json"

DEFAULT_TOP_K = 5
DEFAULT_RELEVANCE_THRESHOLD = 0.25
THRESHOLD_COMPARISON_VALUES = [0.25, 0.30]


def load_labelled_queries() -> list[dict]:
    """Load curated retrieval test cases with expected source filenames."""
    with LABELLED_QUERIES_PATH.open("r", encoding="utf-8") as query_file:
        return json.load(query_file)


def source_filename_from_chunk(chunk) -> str:
    """Extract only the filename from a retrieved chunk source path."""
    source_path = chunk.metadata.get("source", "")
    return Path(source_path).name


def format_retrieved_chunk(document, score: float) -> dict:
    """Convert one scored retrieved chunk into evaluation-friendly output."""
    source_filename = source_filename_from_chunk(document)

    return {
        "source": source_filename,
        "score": round(float(score), 4),
    }


def get_unique_sources(retrieved_chunks: list[dict]) -> list[str]:
    """Return unique retrieved source filenames while preserving retrieval order."""
    return list(
        dict.fromkeys(
            retrieved_chunk["source"]
            for retrieved_chunk in retrieved_chunks
        )
    )


def evaluate_permission_block_case(test_case: dict) -> dict:
    """Check whether a restricted matching document is blocked before answer generation."""
    result = generate_answer(
        question=test_case["question"],
        role=test_case["role"],
        department=test_case["department"],
        department_filter=test_case.get("department_filter"),
        file_type_filter=test_case.get("file_type_filter"),
    )

    answer = result["answer"]
    permission_blocked = "insufficient permission" in answer.lower()

    return {
        "query_id": test_case["query_id"],
        "suite": test_case.get("suite", "default"),
        "question": test_case["question"],
        "expected_behavior": "permission_block",
        "expected_source": test_case.get("expected_source"),
        "retrieved_sources": result["sources"],
        "retrieved_chunks": [],
        "hit": permission_blocked,
        "issue": "" if permission_blocked else "Expected permission block was not returned.",
    }


def evaluate_retrieval_case(
    test_case: dict,
    minimum_relevance_threshold: float,
) -> dict:
    """Run one labelled retrieval query and compare the result with its expectation."""
    scored_chunks = retrieve_relevant_chunks_with_scores(
        question=test_case["question"],
        role=test_case["role"],
        department=test_case["department"],
        department_filter=test_case.get("department_filter"),
        file_type_filter=test_case.get("file_type_filter"),
        top_k=DEFAULT_TOP_K,
        minimum_relevance_score=minimum_relevance_threshold,
    )

    retrieved_chunks = [
        format_retrieved_chunk(document, score)
        for document, score in scored_chunks
    ]

    retrieved_sources = get_unique_sources(retrieved_chunks)
    expected_behavior = test_case.get("expected_behavior", "hit")
    expected_source = test_case.get("expected_source")

    if expected_behavior == "miss":
        hit = not retrieved_sources
        issue = "" if hit else "Expected no accepted retrieval, but chunks passed the threshold."
    else:
        hit = expected_source in retrieved_sources
        issue = "" if hit else "Expected source was not found in retrieved top-K chunks."

    return {
        "query_id": test_case["query_id"],
        "suite": test_case.get("suite", "default"),
        "question": test_case["question"],
        "expected_behavior": expected_behavior,
        "expected_source": expected_source,
        "retrieved_sources": retrieved_sources,
        "retrieved_chunks": retrieved_chunks,
        "minimum_relevance_threshold": minimum_relevance_threshold,
        "hit": hit,
        "issue": issue,
    }


def evaluate_query(
    test_case: dict,
    minimum_relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> dict:
    """Evaluate one labelled case as retrieval, expected miss, or permission block."""
    expected_behavior = test_case.get("expected_behavior", "hit")

    if expected_behavior == "permission_block":
        return evaluate_permission_block_case(test_case)

    return evaluate_retrieval_case(
        test_case,
        minimum_relevance_threshold,
    )


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


def build_threshold_comparison(test_cases: list[dict]) -> dict:
    """Compare retrieval pass/miss behavior across candidate relevance thresholds."""
    comparison = {}

    retrieval_cases = [
        test_case
        for test_case in test_cases
        if test_case.get("expected_behavior", "hit") != "permission_block"
    ]

    for threshold in THRESHOLD_COMPARISON_VALUES:
        threshold_results = [
            evaluate_retrieval_case(test_case, threshold)
            for test_case in retrieval_cases
        ]

        comparison[str(threshold)] = {
            "summary": summarize_results(threshold_results),
            "results": threshold_results,
        }

    return comparison


def interpret_threshold_comparison(threshold_comparison: dict) -> dict:
    """Explain whether candidate thresholds change labelled retrieval outcomes."""
    threshold_summaries = {
        threshold: comparison_result["summary"]["overall"]
        for threshold, comparison_result in threshold_comparison.items()
    }

    best_threshold = max(
        threshold_summaries,
        key=lambda threshold: (
            threshold_summaries[threshold]["correct_queries"],
            threshold_summaries[threshold]["top_k_accuracy"],
        ),
    )

    comparison_rows = []

    for threshold, comparison_result in threshold_comparison.items():
        failed_query_ids = [
            result["query_id"]
            for result in comparison_result["results"]
            if not result["hit"]
        ]

        comparison_rows.append(
            {
                "threshold": float(threshold),
                "correct_queries": comparison_result["summary"]["overall"]["correct_queries"],
                "total_queries": comparison_result["summary"]["overall"]["total_queries"],
                "top_k_accuracy_percent": comparison_result["summary"]["overall"]["top_k_accuracy_percent"],
                "failed_query_ids": failed_query_ids,
            }
        )

    unique_accuracy_values = {
        row["top_k_accuracy_percent"]
        for row in comparison_rows
    }

    if len(unique_accuracy_values) == 1:
        recommendation = (
            "Current labelled set shows no accuracy difference between candidate "
            "thresholds. Keep the lower threshold only as a cautious local setting "
            "until more borderline PDF/DOCX cases are added."
        )
    else:
        recommendation = (
            f"Threshold {best_threshold} performs best on the current labelled set. "
            "Review failed query IDs before changing production defaults."
        )

    return {
        "best_threshold": float(best_threshold),
        "comparison_rows": comparison_rows,
        "recommendation": recommendation,
    }


def run_evaluation() -> dict:
    """Run all labelled retrieval cases and calculate Top-K Accuracy."""
    test_cases = load_labelled_queries()

    results = [
        evaluate_query(
            test_case,
            minimum_relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD,
        )
        for test_case in test_cases
    ]

    summary = summarize_results(results)

    miss_rows = [
        result
        for result in results
        if not result["hit"]
    ]

    threshold_comparison = build_threshold_comparison(test_cases)

    return {
        "summary": summary,
        "miss_rows": miss_rows,
        "results": results,
        "threshold_comparison": build_threshold_comparison(test_cases),
        "threshold_interpretation": interpret_threshold_comparison(threshold_comparison),
    }



if __name__ == "__main__":
    evaluation_output = run_evaluation()

    with RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(evaluation_output, results_file, indent=2)

    print(json.dumps(evaluation_output["summary"], indent=2))
    print(f"Saved full results to {RESULTS_PATH}")