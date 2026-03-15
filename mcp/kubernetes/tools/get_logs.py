from typing import Dict, List, TYPE_CHECKING

from kubernetes.client import exceptions as k8s_exceptions

try:
    from ..kube_client import core_v1
except ImportError:  # pragma: no cover
    from kube_client import core_v1  # type: ignore

from .observability_helpers import greedy_compress_lines

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def get_logs(namespace: str, service: str) -> List[Dict[str, object]]:
        """Collect log data from a pod using Kubernetes APIs."""
        label_selector = _service_selector(namespace, service)
        if not label_selector:
            try:
                svc = core_v1().read_namespaced_service(name=service, namespace=namespace)
                selector = svc.spec.selector or {}
                label_selector = ",".join(f"{key}={value}" for key, value in selector.items())
            except k8s_exceptions.ApiException as exc:
                if exc.status == 404:
                    return [{"error": "Service or namespace not found. Use kubectl to check."}]
                return [{"error": f"Failed to read service: {exc.reason}"}]

        if not label_selector:
            return [{"error": f"Service '{service}' in namespace '{namespace}' has no selector"}]

        try:
            pods = core_v1().list_namespaced_pod(namespace=namespace, label_selector=label_selector).items
        except k8s_exceptions.ApiException as exc:
            return [{"error": f"Failed to list pods for service '{service}': {exc.reason}"}]

        results: List[Dict[str, object]] = []
        for pod in pods:
            pod_name = pod.metadata.name
            try:
                logs = core_v1().read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    tail_lines=200,
                )
                results.append({"pod": pod_name, "logs": greedy_compress_lines(logs)})
            except k8s_exceptions.ApiException as exc:
                results.append(
                    {
                        "pod": pod_name,
                        "error": f"Failed to read logs: {exc.reason}",
                        "status_code": exc.status,
                    }
                )

        return results


def _service_selector(namespace: str, service: str) -> str:
    if namespace == "test-social-network":
        return f"app={service}"
    if namespace == "test-hotel-reservation":
        return f"io.kompose.service={service}"
    if namespace == "astronomy-shop":
        return f"app.kubernetes.io/name={service}"
    if namespace == "default" and "wrk2-job" in service:
        return "job-name=wrk2-job"
    return ""
