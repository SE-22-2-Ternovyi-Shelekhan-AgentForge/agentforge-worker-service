from typing import Annotated, Any, TypedDict
from uuid import UUID

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    session_id: UUID
    conversation_id: UUID
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    next_agent: str | None
    last_reasoning: str | None
    scratchpad: dict[str, str]
    trace: list[dict[str, Any]]
