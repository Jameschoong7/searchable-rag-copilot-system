import os

from langchain_community.llms import Ollama
from openai import OpenAI

from src.core.config import (
    AZURE_OPENAI_LLM_BACKEND,
    OLLAMA_LLM_BACKEND,
    read_app_config,
)
from src.core.llm_usage_repository import (
    estimate_openai_cost_usd,
    record_llm_usage,
)


class FoundryOpenAIModel:
    """Small adapter so Azure Foundry can be called like the local Ollama model."""

    def __init__(self) -> None:
        """Create the OpenAI-compatible client for the Foundry endpoint."""
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not configured.")

        if not api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not configured.")

        if not deployment:
            raise RuntimeError("AZURE_OPENAI_CHAT_DEPLOYMENT is not configured.")

        self.deployment = deployment
        self.client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )

    def invoke(self, prompt: str, operation: str = "chat") -> str:
        """Send one prompt to the configured Foundry deployment and return text."""
        response = self.client.responses.create(
            model=self.deployment,
            input=prompt,
        )
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        estimated_cost = estimate_openai_cost_usd(
            deployment=self.deployment,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        record_llm_usage(
            backend=AZURE_OPENAI_LLM_BACKEND,
            deployment=self.deployment,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )

        return response.output_text


def create_chat_llm():
    """Return the configured LLM while preserving the existing invoke(prompt) API."""
    config = read_app_config()

    if config.llm_backend == OLLAMA_LLM_BACKEND:
        return Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("OLLAMA_MODEL"),
            temperature=0,
        )

    if config.llm_backend == AZURE_OPENAI_LLM_BACKEND:
        return FoundryOpenAIModel()

    raise RuntimeError(f"Unsupported LLM_BACKEND: {config.llm_backend}")


def invoke_configured_llm(llm, prompt: str, operation: str = "chat") -> str:
    """Invoke a configured LLM with usage labels when the backend supports them."""
    try:
        return llm.invoke(prompt, operation=operation)
    except TypeError:
        return llm.invoke(prompt)
