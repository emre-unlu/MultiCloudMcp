"""MCP tool for listing Kubernetes nodes with readiness info."""

from typing import Dict, List, TYPE_CHECKING

try:
    from ..kube_client import core_v1
except ImportError:  # pragma: no cover
    from kube_client import core_v1  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def list_nodes() -> List[Dict[str, object]]:
        """List cluster nodes along with readiness state and roles."""
        nodes = core_v1().list_node().items
        summaries: List[Dict[str, object]] = []
        for node in nodes:
            conditions = node.status.conditions or []
            ready_condition = next((c for c in conditions if c.type == "Ready"), None)
            ready = ready_condition.status == "True" if ready_condition else False
            roles = [
                label.replace("node-role.kubernetes.io/", "")
                for label, value in (node.metadata.labels or {}).items()
                if label.startswith("node-role.kubernetes.io/") and value in ("", "true")
            ]
            summaries.append(
                {
                    "name": node.metadata.name,
                    "ready": ready,
                    "roles": roles,
                    "taints": [
                        {"key": t.key, "value": t.value, "effect": t.effect}
                        for t in (node.spec.taints or [])
                    ],
                    "kubelet_version": getattr(node.status, "node_info", None).kubelet_version
                    if getattr(node.status, "node_info", None)
                    else None,
                }
            )
        return summaries
