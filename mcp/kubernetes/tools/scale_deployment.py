from typing import Dict, TYPE_CHECKING

from kubernetes.client import exceptions as k8s_exceptions

try:
    from ..kube_client import apps_v1
except ImportError:  # pragma: no cover
    from kube_client import apps_v1  # type: ignore

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def scale_deployment(namespace: str, name: str, replicas: int) -> Dict:
        """Scale an existing deployment (replicas 1..5)."""
        if replicas < 1 or replicas > 5:
            raise ValueError("replicas must be between 1 and 5 for demo safety")
        try:
            res = apps_v1().patch_namespaced_deployment_scale(
                name,
                namespace,
                {"spec": {"replicas": replicas}},
            )
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                raise ValueError(f"Deployment '{name}' in namespace '{namespace}' not found") from exc
            raise
        return {"name": name, "namespace": namespace, "replicas": res.spec.replicas}
