from typing import List, TYPE_CHECKING

try:
    from ..kube_client import core_v1
except ImportError:  # pragma: no cover
    from kube_client import core_v1  # type: ignore

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def list_namespaces() -> List[str]:
        """List Kubernetes namespaces."""
        return [ns.metadata.name for ns in core_v1().list_namespace().items]
