"""Expose pod event retrieval through MCP."""

from typing import Dict, TYPE_CHECKING

from .diagnostics import KubernetesDiagnostics

if TYPE_CHECKING:  # pragma: no cover
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    diagnostics = KubernetesDiagnostics()

    @mcp.tool()
    def pod_events(namespace: str, pod: str) -> Dict[str, object]:
        """List recent Kubernetes events for a pod."""
        return diagnostics.get_pod_events(pod_name=pod, namespace=namespace)
