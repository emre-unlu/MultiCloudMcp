from typing import Dict, List, TYPE_CHECKING

from kubernetes.client import exceptions as k8s_exceptions

try:
    from ..kube_client import core_v1
except ImportError:  # pragma: no cover
    from kube_client import core_v1  # type: ignore

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def list_pods(namespace: str) -> List[Dict[str, object]]:
        """List pods in a namespace with status details."""
        try:
            pods = core_v1().list_namespaced_pod(namespace).items
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                raise ValueError(f"Namespace '{namespace}' not found") from exc
            raise

        pod_summaries: List[Dict[str, object]] = []
        for pod in pods:
            statuses = pod.status.container_statuses or []
            ready = sum(1 for status in statuses if status.ready)
            total = len(statuses)
            restarts = sum(status.restart_count for status in statuses)

            container_states = []
            for status in statuses:
                state = status.state
                waiting = getattr(state, "waiting", None)
                terminated = getattr(state, "terminated", None)
                running = getattr(state, "running", None)

                if waiting:
                    state_info = {"state": "waiting", "reason": waiting.reason, "message": waiting.message}
                elif terminated:
                    state_info = {
                        "state": "terminated",
                        "reason": terminated.reason,
                        "exit_code": terminated.exit_code,
                        "message": terminated.message,
                    }
                elif running:
                    state_info = {"state": "running", "started_at": running.started_at.isoformat() if running.started_at else None}
                else:
                    state_info = {"state": "unknown"}

                container_states.append(
                    {
                        "name": status.name,
                        "ready": status.ready,
                        "restart_count": status.restart_count,
                        "state": state_info,
                    }
                )

            pod_summaries.append(
                {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "ready": f"{ready}/{total}",
                    "restarts": restarts,
                    "node": pod.spec.node_name,
                    "reason": pod.status.reason,
                    "containers": container_states,
                }
            )

        return pod_summaries
