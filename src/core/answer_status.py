"""Classify RAG answers for UI labels and query-log metrics."""


def classify_answer_status(answer: str, sources: list[str]) -> str:
    """Return the user-facing outcome status for one generated answer."""
    lowered_answer = answer.lower()

    if "insufficient permission" in lowered_answer:
        return "permission_block"

    not_found_phrases = [
        "not explicitly stated",
        "unable to find",
        "information missing",
        "could not find",
        "not found",
        "provided sources do not contain",
        "sources do not contain",
        "available documents do not contain",
        "not contain information about",
        "not supported by the provided source",
        "not supported by the provided sources",
    ]

    if any(phrase in lowered_answer for phrase in not_found_phrases):
        return "not_found"

    if not sources:
        return "not_found"

    return "success"
