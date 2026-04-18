from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system"]


class ContextEntry(BaseModel):
    from_agent_role: str = Field(
        description="Role of the agent that produced this message (e.g. 'planner', 'coder'). "
        "Prefixed into the LLM prompt so downstream agents can tell who spoke."
    )
    content: str
    message_type: MessageRole = "assistant"


class AgentTaskRequested(BaseModel):
    task_id: UUID
    conversation_id: UUID = Field(
        description="Groups tasks belonging to the same multi-agent conversation. "
        "Worker is stateless; the orchestrator uses this to stitch steps together."
    )
    agent_role: str = Field(
        description="Role of the agent executing this task. Informational on the worker side — "
        "echoed back in the result so the orchestrator knows who answered."
    )
    system_prompt: str
    user_prompt: str
    context: list[ContextEntry] = Field(default_factory=list)
    model: str | None = None
    temperature: float | None = None
