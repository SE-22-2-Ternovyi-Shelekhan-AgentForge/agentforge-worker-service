from uuid import uuid4

from agentforge_worker.tools import ToolContext
from agentforge_worker.tools.current_time import build_current_time
from agentforge_worker.tools.scratchpad import build_scratchpad_reader, build_scratchpad_writer


def _ctx(scratchpad=None) -> ToolContext:
    return ToolContext(session_id=uuid4(), scratchpad=scratchpad if scratchpad is not None else {})


def test_current_time_returns_iso_string():
    tool = build_current_time(_ctx())
    result = tool.invoke({})
    assert "T" in result
    assert "+" in result or "Z" in result or result.endswith("+00:00")


def test_scratchpad_writer_stores_value():
    pad: dict = {}
    ctx = _ctx(scratchpad=pad)
    writer = build_scratchpad_writer(ctx)

    result = writer.invoke({"key": "summary", "value": "hello world"})

    assert pad["summary"] == "hello world"
    assert "summary" in result


def test_scratchpad_reader_returns_value():
    pad = {"note": "important finding"}
    ctx = _ctx(scratchpad=pad)
    reader = build_scratchpad_reader(ctx)

    result = reader.invoke({"key": "note"})
    assert result == "important finding"


def test_scratchpad_reader_missing_key_returns_not_found():
    ctx = _ctx(scratchpad={})
    reader = build_scratchpad_reader(ctx)

    result = reader.invoke({"key": "nonexistent"})
    assert result == "NOT_FOUND"
