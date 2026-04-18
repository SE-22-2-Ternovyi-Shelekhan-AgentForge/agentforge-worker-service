from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskCompletedEvent(BaseModel):
    task_id: UUID
    conversation_id: UUID
    agent_role: str
    output: str
    model_used: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    completed_at: datetime = Field(default_factory=_utc_now)


class TaskFailedEvent(BaseModel):
    task_id: UUID
    conversation_id: UUID
    agent_role: str
    error_type: str
    error_message: str
    failed_at: datetime = Field(default_factory=_utc_now)
