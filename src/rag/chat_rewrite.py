import os
import re

from langchain_community.llms import Ollama


MAX_REWRITE_HISTORY_MESSAGES = 6


def normalise_message_role(role: str) -> str:
    """Return a readable role label for the rewrite prompt."""
    if role == "assistant":
        return "Assistant"

    return "User"


def get_recent_messages(messages: list[dict]) -> list[dict]:
    """Return the bounded recent context window used for follow-up rewriting."""
    return messages[-MAX_REWRITE_HISTORY_MESSAGES:]


def build_rewrite_prompt(question: str, recent_messages: list[dict]) -> str:
    """Build a bounded prompt that asks the LLM for a standalone retrieval query."""
    history_lines = []

    for message in get_recent_messages(recent_messages):
        role = normalise_message_role(message.get("message_role", "user"))
        content = " ".join(message.get("content", "").split())

        if content:
            history_lines.append(f"{role}: {content}")

    history_text = "\n".join(history_lines)

    return f"""
    Rewrite the latest user question into one standalone search query.

    Rules:
    - Use the recent conversation only to resolve references such as "it", "that", "step 2", or "the second one".
    - Do not answer the question.
    - Do not add facts that are not present in the conversation.
    - Return only the rewritten standalone question.
    - If the latest question is already standalone, return it unchanged.

    Recent conversation:
    {history_text}

    Latest user question:
    {question.strip()}
    """.strip()


def clean_rewrite_output(output: str, fallback_question: str) -> str:
    """Normalize the LLM rewrite and fall back if it produced unusable text."""
    cleaned = output.strip().strip('"').strip("'")
    cleaned = re.sub(
        r"(?i)^standalone\s+(question|query)\s*:\s*",
        "",
        cleaned,
    ).strip()

    if not cleaned:
        return fallback_question.strip()

    if len(cleaned) > 300:
        return fallback_question.strip()

    return cleaned


def rewrite_follow_up_question(
    question: str,
    recent_messages: list[dict],
    rewrite_callable,
) -> str:
    """Rewrite a follow-up into a standalone retrieval query using bounded history."""
    clean_question = question.strip()

    if not recent_messages:
        return clean_question

    prompt = build_rewrite_prompt(clean_question, recent_messages)
    rewritten = rewrite_callable(prompt)

    return clean_rewrite_output(str(rewritten), clean_question)


def rewrite_follow_up_question_with_ollama(
    question: str,
    recent_messages: list[dict],
) -> str:
    """Use the configured local LLM to rewrite a follow-up, falling back safely."""
    if not recent_messages:
        return question.strip()

    try:
        llm = Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("OLLAMA_MODEL"),
            temperature=0,
        )
        return rewrite_follow_up_question(
            question,
            recent_messages,
            llm.invoke,
        )
    except Exception:
        return question.strip()
