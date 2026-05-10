from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool


def build_web_search(_ctx) -> BaseTool:  # noqa: ANN001 — context unused for stateless tool
    tool = DuckDuckGoSearchRun()
    tool.name = "web_search"
    tool.description = (
        "Search the public web via DuckDuckGo. "
        "Input: a free-form search query string. "
        "Returns: a short snippet aggregating the top results."
    )
    return tool
