import os

from langchain_community.llms import Ollama
from openai import OpenAI

from src.core.config import (
    AZURE_OPENAI_LLM_BACKEND,
    OLLAMA_LLM_BACKEND,
    read_app_config,
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

    def invoke(self, prompt: str) -> str:
        """Send one prompt to the configured Foundry deployment and return text."""
        response = self.client.responses.create(
            model=self.deployment,
            input=prompt,
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
