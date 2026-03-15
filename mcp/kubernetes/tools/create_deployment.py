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
    def create_deployment(namespace: str, name: str, image: str = "nginx", replicas: int = 1) -> Dict:
        """Create a small deployment (replicas 1..5)."""
        if replicas < 1 or replicas > 5:
            raise ValueError("replicas must be between 1 and 5 for demo safety")

        body = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": name},
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": {"app": name}},
                "template": {
                    "metadata": {"labels": {"app": name}},
                    "spec": {
                        "containers": [
                            {
                                "name": name,
                                "image": image,
                                "ports": [{"containerPort": 80}],
                            }
                        ]
                    },
                },
            },
        }

        try:
            res = apps_v1().create_namespaced_deployment(namespace, body)
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                raise ValueError(f"Namespace '{namespace}' not found") from exc
            raise

        return {"name": res.metadata.name, "namespace": namespace, "replicas": res.spec.replicas}
