from datetime import datetime, timezone

from langchain_core.tools import BaseTool, tool


def build_current_time(_ctx) -> BaseTool:  # noqa: ANN001 — context unused
    @tool("current_time")
    def current_time() -> str:
        """Return the current UTC time in ISO-8601 format. Takes no arguments."""
        return datetime.now(timezone.utc).isoformat()

    return current_time
