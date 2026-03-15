from typing import Optional, TYPE_CHECKING

from kubernetes.client import exceptions as k8s_exceptions

try:
    from ..kube_client import core_v1
except ImportError:  # pragma: no cover
    from kube_client import core_v1  # type: ignore

from .diagnostics import KubernetesDiagnostics

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def pod_logs(namespace: str, pod: str, tail_lines: int = 80) -> str:
        """Get the last N log lines for a pod."""
        try:
            return core_v1().read_namespaced_pod_log(
                name=pod,
                namespace=namespace,
                tail_lines=tail_lines,
            )
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                raise ValueError(f"Pod '{pod}' in namespace '{namespace}' not found") from exc
            details = exc.body or exc.reason or str(exc)
            fallback = None
            try:
                diag = KubernetesDiagnostics()
                events = diag.get_pod_events(pod_name=pod, namespace=namespace)
                if events.get("status") == "success":
                    fallback = events["events"][:5]
            except Exception:
                fallback = None

            return {
                "error": "log_fetch_failed",
                "status_code": exc.status,
                "details": details.strip(),
                "events": fallback,
            }
