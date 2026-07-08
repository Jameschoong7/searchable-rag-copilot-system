import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLM_USAGE_DB_PATH = PROJECT_ROOT / "data/metadata/document_metadata.db"


def now_text() -> str:
    """Return a consistent timestamp string for LLM usage records."""
    return datetime.now().isoformat(timespec="seconds")


def initialise_llm_usage_table() -> None:
    """Create the local LLM usage table if it does not exist."""
    LLM_USAGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(LLM_USAGE_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_records (
                usage_id TEXT PRIMARY KEY,
                backend TEXT NOT NULL,
                deployment TEXT NOT NULL,
                operation TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                estimated_cost_usd REAL,
                created_at TEXT NOT NULL
            )
            """
        )


def estimate_openai_cost_usd(
    deployment: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Estimate token cost for known Foundry/OpenAI demo deployments."""
    if input_tokens is None or output_tokens is None:
        return None

    # Prices are USD per 1M tokens for current demo planning. Keep this estimate
    # visible as approximate admin telemetry, not billing-grade accounting.
    price_table = {
        "nano": {
            "input": 0.20,
            "output": 1.25,
        },
        "mini": {
            "input": 0.75,
            "output": 4.50,
        },
    }
    deployment_name = deployment.lower()
    selected_price = None

    for marker, price in price_table.items():
        if marker in deployment_name:
            selected_price = price
            break

    if selected_price is None:
        return None

    return round(
        (input_tokens / 1_000_000 * selected_price["input"])
        + (output_tokens / 1_000_000 * selected_price["output"]),
        6,
    )


def record_llm_usage(
    backend: str,
    deployment: str,
    operation: str,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    estimated_cost_usd: float | None,
) -> dict:
    """Persist one LLM usage record for admin cost observability."""
    initialise_llm_usage_table()

    record = {
        "usage_id": str(uuid.uuid4()),
        "backend": backend,
        "deployment": deployment,
        "operation": operation,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "created_at": now_text(),
    }

    with sqlite3.connect(LLM_USAGE_DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO llm_usage_records (
                usage_id,
                backend,
                deployment,
                operation,
                input_tokens,
                output_tokens,
                total_tokens,
                estimated_cost_usd,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["usage_id"],
                record["backend"],
                record["deployment"],
                record["operation"],
                record["input_tokens"],
                record["output_tokens"],
                record["total_tokens"],
                record["estimated_cost_usd"],
                record["created_at"],
            ),
        )

    return record


def list_recent_llm_usage(limit: int = 25) -> list[dict]:
    """Return recent LLM usage records for the admin Settings page."""
    initialise_llm_usage_table()

    with sqlite3.connect(LLM_USAGE_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                created_at,
                backend,
                deployment,
                operation,
                input_tokens,
                output_tokens,
                total_tokens,
                estimated_cost_usd
            FROM llm_usage_records
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]
