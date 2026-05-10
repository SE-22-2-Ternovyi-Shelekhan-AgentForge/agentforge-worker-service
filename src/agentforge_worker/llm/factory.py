"""Provider-agnostic LLM factory.

The rest of the worker (supervisor, agent nodes) builds its chat model only
through ``make_chat_model``. Adding a new provider means extending this file —
no changes elsewhere.
"""

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from agentforge_worker.config import Settings


class UnknownProviderError(ValueError):
    """Raised when AgentConfig.provider is not in the supported set."""


def make_chat_model(
    provider: str | None,
    model: str,
    temperature: float,
    timeout: float,
    settings: Settings,
) -> BaseChatModel:
    p = (provider or settings.default_provider).lower()
    if p == "ollama":
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )
    if p == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=timeout,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    raise UnknownProviderError(f"unsupported LLM provider: {provider!r}")
