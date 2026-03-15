from typing import TYPE_CHECKING

from .observability_helpers import read_csv

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def read_metrics(file_path: str) -> str:
        """Reads and returns metrics from a specified CSV file."""
        return read_csv(file_path)
