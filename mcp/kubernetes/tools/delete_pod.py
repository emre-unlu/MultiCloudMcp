from typing import Dict, TYPE_CHECKING, Optional

from kubernetes import client as k8s_client
from kubernetes.client import exceptions as k8s_exceptions

try:
    from ..kube_client import core_v1
except ImportError:  # pragma: no cover
    from kube_client import core_v1  # type: ignore
try:
    from ..kube_client import apps_v1
except ImportError:  # pragma: no cover
    from kube_client import apps_v1  # type: ignore

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def delete_pod(namespace: str, name: str, grace_period_seconds: int | None = None) -> Dict[str, object]:
        """Delete a pod and scale the owning deployment down when applicable."""
        if not namespace:
            raise ValueError("namespace is required")
        if not name:
            raise ValueError("name is required")

        delete_opts = None
        if grace_period_seconds is not None:
            if grace_period_seconds < 0:
                raise ValueError("grace_period_seconds cannot be negative")
            delete_opts = k8s_client.V1DeleteOptions(grace_period_seconds=grace_period_seconds)

        owning_deployment: Optional[str] = None
        pod_obj = None
        try:
            pod_obj = core_v1().read_namespaced_pod(name=name, namespace=namespace)
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                raise ValueError(f"Pod '{name}' in namespace '{namespace}' not found") from exc
            raise

        if pod_obj and pod_obj.metadata and pod_obj.metadata.owner_references:
            for ref in pod_obj.metadata.owner_references:
                if ref.kind == "ReplicaSet" and ref.name:
                    try:
                        rs = apps_v1().read_namespaced_replica_set(name=ref.name, namespace=namespace)
                    except k8s_exceptions.ApiException:
                        continue
                    if rs.metadata and rs.metadata.owner_references:
                        for rs_ref in rs.metadata.owner_references:
                            if rs_ref.kind == "Deployment" and rs_ref.name:
                                owning_deployment = rs_ref.name
                                break
                    if owning_deployment:
                        break

        scaled_info: Optional[Dict[str, object]] = None
        if owning_deployment:
            try:
                deployment = apps_v1().read_namespaced_deployment(
                    name=owning_deployment,
                    namespace=namespace,
                )
                current_replicas = deployment.spec.replicas or 0
            except k8s_exceptions.ApiException:
                current_replicas = None
            if current_replicas is not None and current_replicas > 0:
                new_replicas = max(0, current_replicas - 1)
                if new_replicas != current_replicas:
                    try:
                        apps_v1().patch_namespaced_deployment_scale(
                            name=owning_deployment,
                            namespace=namespace,
                            body={"spec": {"replicas": new_replicas}},
                        )
                        scaled_info = {
                            "deployment": owning_deployment,
                            "replicas": new_replicas,
                        }
                    except k8s_exceptions.ApiException:
                        scaled_info = {
                            "deployment": owning_deployment,
                            "error": "failed_to_scale_deployment",
                        }

        try:
            core_v1().delete_namespaced_pod(
                name=name,
                namespace=namespace,
                body=delete_opts,
                grace_period_seconds=grace_period_seconds,
            )
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                raise ValueError(f"Pod '{name}' in namespace '{namespace}' not found") from exc
            raise

        result: Dict[str, object] = {
            "status": "deleted",
            "namespace": namespace,
            "name": name,
            "kind": "Pod",
        }
        if scaled_info:
            result["scaled"] = scaled_info
        elif owning_deployment:
            result["scaled"] = {
                "deployment": owning_deployment,
                "replicas": "unchanged",
            }
        return result
