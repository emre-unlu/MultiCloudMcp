from typing import Dict, TYPE_CHECKING

from kubernetes import client as k8s_client
from kubernetes.client import exceptions as k8s_exceptions

try:
    from ..kube_client import apps_v1
except ImportError:  # pragma: no cover
    from kube_client import apps_v1  # type: ignore

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def delete_deployment(namespace: str, name: str, grace_period_seconds: int | None = None) -> Dict[str, str]:
        """Delete a deployment after double-checking namespace/name."""
        if not namespace:
            raise ValueError("namespace is required")
        if not name:
            raise ValueError("name is required")

        delete_opts = None
        if grace_period_seconds is not None:
            if grace_period_seconds < 0:
                raise ValueError("grace_period_seconds cannot be negative")
            delete_opts = k8s_client.V1DeleteOptions(grace_period_seconds=grace_period_seconds)

        try:
            apps_v1().delete_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=delete_opts,
                propagation_policy="Foreground",
            )
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                raise ValueError(f"Deployment '{name}' in namespace '{namespace}' not found") from exc
            raise

        return {
            "status": "deleted",
            "namespace": namespace,
            "name": name,
        }
