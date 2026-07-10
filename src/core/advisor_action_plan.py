from src.rag.llm_factory import create_chat_llm, invoke_configured_llm


def build_advisor_action_plan_prompt(recommendation: dict) -> str:
    """Build a guarded prompt for rewriting one advisor row into admin actions."""
    sources_checked = recommendation.get("Sources Checked") or []

    if isinstance(sources_checked, list):
        sources_text = "\n".join(f"- {source}" for source in sources_checked) or "None"
    else:
        sources_text = str(sources_checked) or "None"

    return f"""
You are helping an admin improve an internal RAG knowledge base.

Use only the structured evidence below. The rule-based advisor has already decided
the issue type, priority, reason, owner, and suggested action. Treat those fields
as the source of truth.

Do not answer the user's original question.
Do not invent policy facts, document contents, permissions, or source availability.
Do not recommend bypassing ACL/RBAC rules.
Do not say that a document exists unless it is listed in Sources Checked.

Write a concise admin action plan in 3-5 bullet points.
Make it specific to the query where possible.
Focus on what the admin or document owner should check, update, or measure next.

Structured evidence:
Query: {recommendation.get("Query", "")}
User Scope: {recommendation.get("User Scope", "")}
Issue Type: {recommendation.get("Issue Type", "")}
Priority: {recommendation.get("Priority", "")}
Reason: {recommendation.get("Reason", "")}
Suggested Action: {recommendation.get("Suggested Action", "")}
Owner: {recommendation.get("Owner", "")}
Feedback: {recommendation.get("Feedback", "none")}
Sources Checked:
{sources_text}
""".strip()


def generate_advisor_action_plan(recommendation: dict) -> str:
    """Generate a human-readable admin action plan for one advisor recommendation."""
    prompt = build_advisor_action_plan_prompt(recommendation)
    llm = create_chat_llm()
    action_plan = invoke_configured_llm(
        llm,
        prompt,
        operation="advisor_action_plan",
    )

    return str(action_plan).strip()
