from dataclasses import dataclass
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agentforge_worker.config import settings
from agentforge_worker.contracts import ContextEntry


@dataclass(slots=True)
class LLMResponse:
    output: str
    model_used: str
    tokens_in: int | None
    tokens_out: int | None


def build_chat_model(model: str, temperature: float) -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
    )


def _context_to_message(entry: ContextEntry) -> BaseMessage:
    # Prefix with [<from_agent_role>] so the LLM can tell inter-agent handoffs apart.
    text = f"[{entry.from_agent_role}]: {entry.content}"
    if entry.message_type == "system":
        return SystemMessage(content=text)
    if entry.message_type == "assistant":
        return AIMessage(content=text)
    return HumanMessage(content=text)


def _build_messages(
    system_prompt: str,
    user_prompt: str,
    context: Iterable[ContextEntry],
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    messages.extend(_context_to_message(entry) for entry in context)
    messages.append(HumanMessage(content=user_prompt))
    return messages


def _extract_usage(response: BaseMessage) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None, None
    return usage.get("input_tokens"), usage.get("output_tokens")


def invoke_llm(
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    context: Iterable[ContextEntry],
) -> LLMResponse:
    chat = build_chat_model(model, temperature)
    messages = _build_messages(system_prompt, user_prompt, context)
    response = chat.invoke(messages)
    tokens_in, tokens_out = _extract_usage(response)
    return LLMResponse(
        output=str(response.content),
        model_used=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
