"""Classify RAG answers for UI labels and query-log metrics."""


NOT_FOUND_PHRASES = [
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


def classify_answer_status_detail(answer: str, sources: list[str]) -> dict[str, str]:
    """Return both the answer outcome and the reason for that classification."""
    lowered_answer = answer.lower()

    if "insufficient permission" in lowered_answer:
        return {
            "status": "permission_block",
            "reason": "Insufficient permission response detected",
        }

    for phrase in NOT_FOUND_PHRASES:
        if phrase in lowered_answer:
            return {
                "status": "not_found",
                "reason": f"Grounded refusal phrase detected: {phrase}",
            }

    if not sources:
        return {
            "status": "not_found",
            "reason": "No sources returned from authorised retrieval",
        }

    return {
        "status": "success",
        "reason": "Answer generated with supporting retrieved sources",
    }


def classify_answer_status(answer: str, sources: list[str]) -> str:
    """Return the user-facing outcome status for one generated answer."""
    return classify_answer_status_detail(answer, sources)["status"]
