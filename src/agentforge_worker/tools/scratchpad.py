from langchain_core.tools import BaseTool, tool


def build_scratchpad_writer(ctx) -> BaseTool:  # noqa: ANN001 — ToolContext (avoid circular import)
    pad = ctx.scratchpad

    @tool("write_scratchpad")
    def write_scratchpad(key: str, value: str) -> str:
        """Store a string value under a key in the team's shared scratchpad for this session. Input: key (str), value (str). Returns confirmation."""
        pad[key] = value
        return f"stored key='{key}' ({len(value)} chars)"

    return write_scratchpad


def build_scratchpad_reader(ctx) -> BaseTool:  # noqa: ANN001
    pad = ctx.scratchpad

    @tool("read_scratchpad")
    def read_scratchpad(key: str) -> str:
        """Read a value previously written to the team's shared scratchpad by another agent. Input: key (str). Returns the value or 'NOT_FOUND'."""
        if key not in pad:
            return "NOT_FOUND"
        return pad[key]

    return read_scratchpad
