from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TraceEntry(BaseModel):
    agent_role: str
    output: str
    tools_used: list[str] = Field(default_factory=list)
    tokens_in: int | None = None
    tokens_out: int | None = None


ErrorType = Literal[
    "llm_timeout",
    "llm_connection_error",
    "llm_api_error",
    "tool_error",
    "parse_error",
    "max_iterations_exceeded",
    "unknown",
]


class AgentSessionCompleted(BaseModel):
    session_id: UUID
    conversation_id: UUID
    final_output: str
    trace: list[TraceEntry] = Field(default_factory=list)
    tokens_in_total: int | None = None
    tokens_out_total: int | None = None
    iterations: int
    completed_at: datetime = Field(default_factory=_utc_now)


class AgentSessionFailed(BaseModel):
    session_id: UUID
    conversation_id: UUID
    error_type: ErrorType
    error_message: str
    partial_trace: list[TraceEntry] = Field(default_factory=list)
    failed_at: datetime = Field(default_factory=_utc_now)
