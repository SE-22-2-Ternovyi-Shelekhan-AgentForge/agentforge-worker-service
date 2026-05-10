from dataclasses import dataclass, field
from typing import Callable
from uuid import UUID

from langchain_core.tools import BaseTool

from agentforge_worker.tools.current_time import build_current_time
from agentforge_worker.tools.read_file import build_read_file
from agentforge_worker.tools.scratchpad import build_scratchpad_reader, build_scratchpad_writer
from agentforge_worker.tools.web_search import build_web_search


@dataclass
class ToolContext:
    session_id: UUID
    scratchpad: dict[str, str] = field(default_factory=dict)


ToolFactory = Callable[[ToolContext], BaseTool]

_REGISTRY: dict[str, ToolFactory] = {
    "web_search": build_web_search,
    "current_time": build_current_time,
    "read_file": build_read_file,
    "write_scratchpad": build_scratchpad_writer,
    "read_scratchpad": build_scratchpad_reader,
}


def get_tools(names: list[str], context: ToolContext) -> list[BaseTool]:
    return [_REGISTRY[n](context) for n in names if n in _REGISTRY]
