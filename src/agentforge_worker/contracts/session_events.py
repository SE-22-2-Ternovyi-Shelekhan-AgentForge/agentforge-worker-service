from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


EventType = Literal[
    "session_started",
    "supervisor_routed",
    "agent_started",
    "tool_called",
    "tool_result",
    "agent_finished",
    "session_finished",
]


class AgentEventOccurred(BaseModel):
    session_id: UUID
    conversation_id: UUID
    event_type: EventType
    agent_role: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utc_now)
