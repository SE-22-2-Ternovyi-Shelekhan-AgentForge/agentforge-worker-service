from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ContextMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    agent_role: str | None = None
    timestamp: datetime | None = None


class AgentConfig(BaseModel):
    role: str
    system_prompt: str
    provider: str | None = None        # "ollama" | "openai"; None → worker default
    model: str | None = None
    temperature: float | None = None
    tools: list[str] = Field(default_factory=list)


class TeamConfig(BaseModel):
    supervisor_prompt: str | None = None
    agents: list[AgentConfig] = Field(min_length=1)
    max_iterations: int = 10


class AgentSessionRequested(BaseModel):
    session_id: UUID
    conversation_id: UUID
    user_prompt: str
    history: list[ContextMessage] = Field(default_factory=list)
    team: TeamConfig
