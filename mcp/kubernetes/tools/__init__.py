"""
Tool registration helpers for the Kubernetes MCP server.
"""

from typing import TYPE_CHECKING, Callable, List

from . import (
    create_deployment,
    diagnostics,
    delete_deployment,
    delete_pod,
    exec_shell,
    get_logs,
    get_metrics,
    get_traces,
    list_namespaces,
    list_nodes,
    list_pods,
    pod_events,
    pod_logs,
    read_metrics,
    read_traces,
    scale_deployment,
    
)

if TYPE_CHECKING:
    from fastmcp import FastMCP

_REGISTRARS: List[Callable[["FastMCP"], None]] = [
    list_namespaces.register,
    list_nodes.register,
    list_pods.register,
    pod_logs.register,
    pod_events.register,
    create_deployment.register,
    scale_deployment.register,
    get_logs.register,
    get_metrics.register,
    get_traces.register,
    read_metrics.register,
    read_traces.register,
    diagnostics.register,
    delete_deployment.register,
    delete_pod.register,
    exec_shell.register,
]


def register_all(mcp: "FastMCP") -> None:
    for register in _REGISTRARS:
        register(mcp)
