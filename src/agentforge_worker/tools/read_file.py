from pathlib import Path

from langchain_core.tools import BaseTool, tool

from agentforge_worker.config import settings

_MAX_BYTES = 8000


def _safe_read(rel: str) -> str:
    safe_root = Path(settings.workspace_path).resolve()
    target = (safe_root / rel).resolve()
    if safe_root != target and safe_root not in target.parents:
        raise ValueError(f"path '{rel}' escapes workspace root")
    if not target.exists():
        raise FileNotFoundError(f"file '{rel}' not found in workspace")
    if not target.is_file():
        raise ValueError(f"path '{rel}' is not a regular file")
    return target.read_text(encoding="utf-8", errors="replace")[:_MAX_BYTES]


def build_read_file(_ctx) -> BaseTool:  # noqa: ANN001 — context unused
    @tool("read_file")
    def read_file(path: str) -> str:
        """Read a UTF-8 text file from the shared workspace. Input: relative path inside the workspace. Output is capped at 8000 chars."""
        try:
            return _safe_read(path)
        except (ValueError, FileNotFoundError, OSError) as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    return read_file
