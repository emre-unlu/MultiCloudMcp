

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kubernetes.client import exceptions as k8s_exceptions

try:
    from ..kube_client import apps_v1, core_v1, custom_objects
except ImportError:  # pragma: no cover
    from kube_client import apps_v1, core_v1, custom_objects  # type: ignore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastmcp import FastMCP


class KubernetesDiagnostics:
    

    def __init__(self) -> None:
        """Initialize Kubernetes clients."""
        try:
            self.core_v1 = core_v1()
            self.apps_v1 = apps_v1()
            self._custom_objects = None
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to initialize Kubernetes clients: %s", exc)
            raise

    def get_pod_logs(
        self,
        pod_name: str,
        namespace: str = "default",
        container: Optional[str] = None,
        tail_lines: int = 100,
        since_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get logs from a pod."""
        try:
            logs = self.core_v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                container=container,
                tail_lines=tail_lines,
                since_seconds=since_seconds,
            )
            return {
                "status": "success",
                "pod_name": pod_name,
                "namespace": namespace,
                "container": container,
                "logs": logs,
            }
        except k8s_exceptions.ApiException as exc:
            return {"status": "error", "error": f"Failed to get pod logs: {exc.reason}"}

    def analyze_pod_logs(
        self,
        pod_name: str,
        namespace: str = "default",
        container: Optional[str] = None,
        tail_lines: int = 1000,
    ) -> Dict[str, Any]:
        """Analyze pod logs for patterns and issues."""
        logs = self.get_pod_logs(pod_name, namespace, container, tail_lines)
        if logs["status"] != "success":
            return logs

        log_content = logs["logs"]
        analysis = {
            "error_count": 0,
            "warning_count": 0,
            "error_patterns": [],
            "warning_patterns": [],
            "common_patterns": [],
            "time_analysis": self._analyze_log_timing(log_content),
        }

        lines = log_content.split("\n")
        pattern_counts: Dict[str, int] = {}
        for line in lines:
            if re.search(r"error|exception|fail", line, re.IGNORECASE):
                analysis["error_count"] += 1
                analysis["error_patterns"].append(line)
            elif re.search(r"warn|warning", line, re.IGNORECASE):
                analysis["warning_count"] += 1
                analysis["warning_patterns"].append(line)

            pattern = self._extract_log_pattern(line)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        analysis["common_patterns"] = [{"pattern": patt, "count": count} for patt, count in sorted_patterns[:5]]

        return {"status": "success", "pod_name": pod_name, "namespace": namespace, "analysis": analysis}

    def get_pod_events(self, pod_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get events related to a pod."""
        try:
            field_selector = f"involvedObject.name={pod_name}"
            events = self.core_v1.list_namespaced_event(namespace=namespace, field_selector=field_selector)
            formatted = []
            for event in events.items:
                formatted.append(
                    {
                        "type": event.type,
                        "reason": event.reason,
                        "message": event.message,
                        "count": event.count,
                        "first_timestamp": event.first_timestamp.isoformat() if event.first_timestamp else None,
                        "last_timestamp": event.last_timestamp.isoformat() if event.last_timestamp else None,
                    }
                )
            return {"status": "success", "pod_name": pod_name, "namespace": namespace, "events": formatted}
        except k8s_exceptions.ApiException as exc:
            return {"status": "error", "error": f"Failed to get pod events: {exc.reason}"}

    def check_pod_health(self, pod_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Check the health status of a pod."""
        try:
            pod = self.core_v1.read_namespaced_pod(pod_name, namespace)
        except k8s_exceptions.ApiException as exc:
            return {"status": "error", "error": f"Failed to check pod health: {exc.reason}"}

        health_check: Dict[str, Any] = {
            "status": pod.status.phase,
            "container_statuses": [],
            "conditions": [],
            "events": [],
            "warnings": [],
        }

        if pod.status.container_statuses:
            for container in pod.status.container_statuses:
                container_status = {
                    "name": container.name,
                    "ready": container.ready,
                    "restart_count": container.restart_count,
                    "state": self._get_container_state(container.state),
                }
                health_check["container_statuses"].append(container_status)
                if container.restart_count > 5:
                    health_check["warnings"].append(
                        f"Container {container.name} has restarted {container.restart_count} times"
                    )

        if pod.status.conditions:
            for condition in pod.status.conditions:
                health_check["conditions"].append(
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message,
                    }
                )

        events = self.get_pod_events(pod_name, namespace)
        if events["status"] == "success":
            health_check["events"] = events["events"]

        return {"status": "success", "pod_name": pod_name, "namespace": namespace, "health_check": health_check}

    def get_resource_usage(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get resource usage metrics for pods."""
        try:
            metrics_client = self._custom_objects_api()
            if namespace:
                metrics = metrics_client.list_namespaced_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="pods",
                )
            else:
                metrics = metrics_client.list_cluster_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    plural="pods",
                )
        except k8s_exceptions.ApiException as exc:
            return {"status": "error", "error": f"Failed to get resource usage metrics: {exc.reason}"}
        except Exception as exc:
            return {"status": "error", "error": f"Failed to get resource usage metrics: {exc}"}

        formatted_metrics = []
        for pod in metrics.get("items", []):
            pod_metrics = {
                "name": pod["metadata"]["name"],
                "namespace": pod["metadata"]["namespace"],
                "containers": [],
            }
            for container in pod.get("containers", []):
                pod_metrics["containers"].append(
                    {
                        "name": container.get("name"),
                        "cpu": container.get("usage", {}).get("cpu"),
                        "memory": container.get("usage", {}).get("memory"),
                    }
                )
            formatted_metrics.append(pod_metrics)

        return {"status": "success", "metrics": formatted_metrics}

    def validate_resources(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Validate resource configurations and usage."""
        try:
            pods = (
                self.core_v1.list_namespaced_pod(namespace=namespace)
                if namespace
                else self.core_v1.list_pod_for_all_namespaces()
            )
        except k8s_exceptions.ApiException as exc:
            return {"status": "error", "error": f"Failed to list pods: {exc.reason}"}

        validation_results = {"resource_issues": [], "recommendations": []}

        for pod in pods.items:
            pod_name = pod.metadata.name
            pod_namespace = pod.metadata.namespace
            for container in pod.spec.containers:
                container_name = container.name
                resources = container.resources

                if not resources or not resources.requests:
                    validation_results["resource_issues"].append(
                        {
                            "pod": pod_name,
                            "namespace": pod_namespace,
                            "container": container_name,
                            "issue": "No resource requests specified",
                        }
                    )
                    validation_results["recommendations"].append(
                        f"Set resource requests for container {container_name} in pod {pod_name}"
                    )

                if not resources or not resources.limits:
                    validation_results["resource_issues"].append(
                        {
                            "pod": pod_name,
                            "namespace": pod_namespace,
                            "container": container_name,
                            "issue": "No resource limits specified",
                        }
                    )
                    validation_results["recommendations"].append(
                        f"Set resource limits for container {container_name} in pod {pod_name}"
                    )

        return {"status": "success", "validation": validation_results}

    def cluster_overview(
        self, namespace: Optional[str] = None, include_metrics: bool = False
    ) -> Dict[str, Any]:
        """Produce a high-level cluster and namespace overview."""
        result: Dict[str, Any] = {"generated_at": datetime.utcnow().isoformat()}
        warnings: List[str] = []

        try:
            nodes = self.core_v1.list_node().items
        except k8s_exceptions.ApiException as exc:
            raise RuntimeError(f"Unable to list cluster nodes: {exc.reason}") from exc

        node_summaries = []
        for node in nodes:
            conditions = node.status.conditions or []
            ready_condition = next((c for c in conditions if c.type == "Ready"), None)
            ready = ready_condition.status == "True" if ready_condition else False
            if not ready:
                reason = ready_condition.reason if ready_condition and ready_condition.reason else "Unknown"
                warnings.append(f"Node {node.metadata.name} not Ready (reason={reason})")

            roles = [
                label.replace("node-role.kubernetes.io/", "")
                for label, value in (node.metadata.labels or {}).items()
                if label.startswith("node-role.kubernetes.io/") and value in ("", "true")
            ]
            node_info = getattr(node.status, "node_info", None)
            node_summaries.append(
                {
                    "name": node.metadata.name,
                    "ready": ready,
                    "roles": roles,
                    "taints": [
                        {"key": t.key, "value": t.value, "effect": t.effect} for t in (node.spec.taints or [])
                    ],
                    "kubelet_version": node_info.kubelet_version if node_info else None,
                }
            )

        result["nodes"] = node_summaries

        if namespace:
            ns_summary = self._summarize_namespace(namespace)
            result["namespace"] = ns_summary
            if ns_summary.get("warnings"):
                warnings.extend(ns_summary["warnings"])

        if include_metrics:
            metrics = self.get_resource_usage(namespace)
            if metrics["status"] == "success":
                result["metrics"] = metrics["metrics"]
            else:
                warnings.append(metrics["error"])

        validation = self.validate_resources(namespace)
        if validation["status"] == "success":
            result["resource_validation"] = validation["validation"]
        else:
            warnings.append(validation["error"])

        if warnings:
            result["warnings"] = warnings

        return result

    def _summarize_namespace(self, namespace: str) -> Dict[str, Any]:
        """Summarize pods within a namespace for diagnostics."""
        try:
            pod_list = self.core_v1.list_namespaced_pod(namespace)
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                raise ValueError(f"Namespace '{namespace}' not found") from exc
            raise

        pods = []
        warnings: List[str] = []
        for pod in pod_list.items:
            statuses = pod.status.container_statuses or []
            ready = sum(1 for status in statuses if status.ready)
            restarts = sum(status.restart_count for status in statuses)
            total = len(statuses)
            phase = pod.status.phase

            container_details: List[Dict[str, Any]] = []
            for status in statuses:
                state_info = self._get_container_state(status.state)
                container_details.append(
                    {
                        "name": status.name,
                        "ready": status.ready,
                        "restart_count": status.restart_count,
                        "state": state_info,
                    }
                )

                state = state_info.get("state")
                reason = state_info.get("reason") or state_info.get("message")
                if state in {"waiting", "terminated"} and reason:
                    warnings.append(
                        f"Pod {pod.metadata.name} container {status.name} state={state} reason={reason}"
                    )

            if phase != "Running" or ready != total:
                warnings.append(
                    f"Pod {pod.metadata.name} phase={phase} ready={ready}/{total} restarts={restarts}"
                )

            pods.append(
                {
                    "name": pod.metadata.name,
                    "phase": phase,
                    "ready_containers": f"{ready}/{total}",
                    "restarts": restarts,
                    "node": pod.spec.node_name,
                    "age": pod.metadata.creation_timestamp.isoformat() if pod.metadata.creation_timestamp else None,
                    "containers": container_details,
                }
            )

        return {"name": namespace, "pods": pods, "warnings": warnings} if warnings else {"name": namespace, "pods": pods}

    def _custom_objects_api(self):
        if not self._custom_objects:
            self._custom_objects = custom_objects()
        return self._custom_objects

    @staticmethod
    def _get_container_state(state) -> Dict[str, Any]:
        """Extract container state information."""
        if state.running:
            return {
                "state": "running",
                "started_at": state.running.started_at.isoformat() if state.running.started_at else None,
            }
        if state.waiting:
            return {
                "state": "waiting",
                "reason": state.waiting.reason,
                "message": state.waiting.message,
            }
        if state.terminated:
            return {
                "state": "terminated",
                "reason": state.terminated.reason,
                "exit_code": state.terminated.exit_code,
                "message": state.terminated.message,
            }
        return {"state": "unknown"}

    @staticmethod
    def _extract_log_pattern(line: str) -> str:
        """Extract a generic pattern from a log line."""
        pattern = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "TIMESTAMP", line)
        pattern = re.sub(r"[a-f0-9]{8,}", "ID", pattern)
        pattern = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "IP", pattern)
        return pattern

    @staticmethod
    def _analyze_log_timing(log_content: str) -> Dict[str, Any]:
        """Analyze timing patterns in logs."""
        timestamps: List[datetime] = []
        pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"

        for line in log_content.split("\n"):
            match = re.search(pattern, line)
            if not match:
                continue
            try:
                timestamps.append(datetime.fromisoformat(match.group(0)))
            except ValueError:
                continue

        if not timestamps:
            return {"message": "No timestamps found in logs"}

        start_time = min(timestamps)
        end_time = max(timestamps)
        hour_counts: Dict[str, int] = {}
        for ts in timestamps:
            hour = ts.strftime("%Y-%m-%d %H:00")
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        return {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": str(end_time - start_time),
            "log_frequency": dict(sorted(hour_counts.items())),
        }


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def diagnose_cluster(namespace: Optional[str] = None, include_metrics: bool = False) -> Dict[str, Any]:
        diagnostics = KubernetesDiagnostics()
        return diagnostics.cluster_overview(namespace=namespace, include_metrics=include_metrics)
