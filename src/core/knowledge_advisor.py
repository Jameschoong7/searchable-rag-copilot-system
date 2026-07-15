import json
import re
from pathlib import Path


def get_meaningful_query_terms(question: str) -> list[str]:
    """Return simple query terms used by the advisor to spot vague requests."""
    ignored_terms = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "how",
        "is",
        "it",
        "of",
        "the",
        "to",
        "what",
        "who",
    }

    return [
        term.lower()
        for term in re.findall(r"[a-zA-Z0-9]+", question)
        if len(term) >= 3 and term.lower() not in ignored_terms
    ]


def normalize_question_for_repeat_detection(question: str) -> str:
    """Normalize a user question enough to spot recurring weak patterns."""
    terms = get_meaningful_query_terms(question)
    return " ".join(terms)


def build_source_metadata_lookup(documents: list[dict]) -> dict[str, dict]:
    """Map source filenames to metadata rows for advisor diagnosis."""
    lookup = {}

    for document in documents:
        filename = document.get("filename")

        if filename:
            lookup[filename] = document
            lookup[f"data/simulated/{filename}"] = document

    return lookup


def extract_logged_sources(sources_json: str) -> list[str]:
    """Parse the JSON source list stored with a query log row."""
    try:
        sources = json.loads(sources_json)
    except json.JSONDecodeError:
        return []

    if not isinstance(sources, list):
        return []

    return [str(source) for source in sources]


def sources_have_ocr_risk(sources: list[str], metadata_lookup: dict[str, dict]) -> bool:
    """Check whether any logged source is associated with OCR or scanned content."""
    risk_terms = ["ocr", "scanned", "image", "visual"]

    for source in sources:
        document = metadata_lookup.get(source) or metadata_lookup.get(Path(source).name)
        visual_status = (document or {}).get("visual_extraction_status", "").lower()

        if any(term in visual_status for term in risk_terms):
            return True

    return False


def build_advisor_recommendation(row, metadata_lookup: dict[str, dict]) -> dict | None:
    """Classify one query outcome into an admin improvement recommendation."""
    timestamp = row[0]
    role = row[2]
    department = row[3]
    question = row[4]
    status = row[7]
    status_reason = row[8] or ""
    sources = extract_logged_sources(row[10])
    feedback = row[12] or "none"
    terms = get_meaningful_query_terms(question)
    user_scope = f"{role} / {department}"

    if status in ["api_error", "connection_error", "error"]:
        return {
            "Time": timestamp,
            "Query": question,
            "User Scope": user_scope,
            "Issue Type": "Backend Reliability",
            "Priority": "High",
            "Reason": status_reason or "The chat request ended with a backend or connection error.",
            "Suggested Action": "Check FastAPI, model backend, vector backend, and recent job logs before retrying.",
            "Owner": "System Admin",
            "Feedback": feedback,
            "Sources Checked": sources,
        }

    if status == "permission_block":
        return {
            "Time": timestamp,
            "Query": question,
            "User Scope": user_scope,
            "Issue Type": "Permission Block",
            "Priority": "Low",
            "Reason": "Relevant restricted knowledge exists, but this role or department is not allowed to use it.",
            "Suggested Action": "No content fix is required unless the document owner decides this information should be shared.",
            "Owner": "Document Owner",
            "Feedback": feedback,
            "Sources Checked": sources,
        }

    if feedback == "reported_issue" and sources:
        return {
            "Time": timestamp,
            "Query": question,
            "User Scope": user_scope,
            "Issue Type": "Possible Retrieval Miss",
            "Priority": "High",
            "Reason": "The answer used sources, but the user reported an issue.",
            "Suggested Action": "Review retrieved sources, expected answer, metadata tags, and labelled evaluation coverage.",
            "Owner": "System Admin",
            "Feedback": feedback,
            "Sources Checked": sources,
        }

    if status == "not_found":
        if len(terms) < 2:
            return {
                "Time": timestamp,
                "Query": question,
                "User Scope": user_scope,
                "Issue Type": "Vague Query",
                "Priority": "Low",
                "Reason": "The question has too little business context for reliable document matching.",
                "Suggested Action": "Add guided prompt examples or ask the user to include a policy, process, department, or task.",
                "Owner": "Knowledge Admin",
                "Feedback": feedback,
                "Sources Checked": sources,
            }

        if sources_have_ocr_risk(sources, metadata_lookup):
            return {
                "Time": timestamp,
                "Query": question,
                "User Scope": user_scope,
                "Issue Type": "OCR Quality",
                "Priority": "High",
                "Reason": "The checked source set includes scanned, image, OCR, or visual-content documents.",
                "Suggested Action": "Review OCR output, replace with a text-based PDF, or add a corrected text version.",
                "Owner": "Knowledge Admin",
                "Feedback": feedback,
                "Sources Checked": sources,
            }

        if not sources:
            return {
                "Time": timestamp,
                "Query": question,
                "User Scope": user_scope,
                "Issue Type": "Missing Knowledge",
                "Priority": "Medium",
                "Reason": "No authorised source was found for this question.",
                "Suggested Action": "Add or sync the missing policy/process document, then create a labelled evaluation case.",
                "Owner": "Knowledge Admin",
                "Feedback": feedback,
                "Sources Checked": sources,
            }

        return {
            "Time": timestamp,
            "Query": question,
            "User Scope": user_scope,
            "Issue Type": "Candidate Retrieval Gap",
            "Priority": "Medium",
            "Reason": "Sources were checked, but the answer still reported missing information.",
            "Suggested Action": "Review chunking, metadata tags, threshold, and whether the expected source is indexed.",
            "Owner": "System Admin",
            "Feedback": feedback,
            "Sources Checked": sources,
        }

    return None


def build_knowledge_advisor_rows(
    recent_outcome_rows: list,
    documents: list[dict],
) -> list[dict]:
    """Build admin recommendations from recent query outcomes."""
    metadata_lookup = build_source_metadata_lookup(documents)
    advisor_rows = []
    repeated_question_groups = {}

    for row in recent_outcome_rows:
        recommendation = build_advisor_recommendation(row, metadata_lookup)

        if recommendation:
            advisor_rows.append(recommendation)

        normalized_question = normalize_question_for_repeat_detection(row[4])
        if not normalized_question:
            continue

        repeated_question_groups.setdefault(normalized_question, []).append(row)

    for question_rows in repeated_question_groups.values():
        if len(question_rows) < 2:
            continue

        latest_row = question_rows[0]
        status_set = {row[7] for row in question_rows}
        feedback_set = {(row[12] or "none") for row in question_rows}
        needs_review = (
            any(status != "success" for status in status_set)
            or "reported_issue" in feedback_set
        )

        if not needs_review:
            continue

        advisor_rows.append(
            {
                "Time": latest_row[0],
                "Query": latest_row[4],
                "User Scope": f"{latest_row[2]} / {latest_row[3]}",
                "Issue Type": "Recurring Issue Pattern",
                "Priority": "Medium",
                "Reason": (
                    f"A similar question pattern appeared {len(question_rows)} times "
                    "recently and still produced weak or reported outcomes."
                ),
                "Suggested Action": (
                    "Add a clearer FAQ entry, improve metadata tags, or create a labelled "
                    "evaluation case for this recurring need."
                ),
                "Owner": "Knowledge Admin",
                "Feedback": ", ".join(sorted(feedback_set)),
                "Sources Checked": [],
            }
        )

    priority_order = {
        "High": 0,
        "Medium": 1,
        "Low": 2,
    }

    return sorted(
        advisor_rows,
        key=lambda item: (priority_order.get(item["Priority"], 9), item["Time"]),
    )
