

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

import requests
from langchain.agents import create_agent
from langchain.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage, HumanMessage

try:
    from langchain.tools import StructuredTool
except ImportError:  
    StructuredTool = None
try:
    from langchain_ollama import ChatOllama
except ImportError:  
    ChatOllama = None

K8S_MCP_URL = os.getenv("K8S_MCP_URL", "http://127.0.0.1:8080")
HDRS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
_SESSION = requests.Session()
CHECKPOINTER = InMemorySaver()
DIAGNOSTICS_CHECKPOINTER = InMemorySaver()
HITL_POLICY: Dict[str, Any] = {
    "k8s_create_deployment": True,
    "k8s_scale_deployment": True,
    "k8s_delete_deployment": True,
    "k8s_delete_pod": True,
    "k8s_list_namespaces": False,
    "k8s_list_nodes": False,
    "k8s_list_pods": False,
    "k8s_pod_logs": False,
    "k8s_pod_events": False,
    "k8s_run_diagnostics": False,
    "k8s_get_namespace": False,
    "k8s_get_logs": False,
    "k8s_get_metrics": False,
    "k8s_get_traces": False,
    "k8s_read_metrics": False,
    "k8s_read_traces": False,
    "k8s_exec_shell": False,
}

DIAGNOSTICS_TOOL_ALLOWLIST = {
    "k8s_list_pods",
    "k8s_pod_events",
    "k8s_pod_logs",
    "k8s_get_logs",
    "k8s_get_metrics",
    "k8s_get_traces",
    "k8s_read_metrics",
    "k8s_read_traces",
    "k8s_exec_shell",
    
}
SUPERVISOR_TOOL_DENYLIST = {
    "k8s_diagnose_cluster",
    "k8s_pod_logs",
    "k8s_get_logs",
    "k8s_get_metrics",
    "k8s_get_traces",
    "k8s_read_metrics",
    "k8s_read_traces",
    "k8s_exec_shell",

}

DIAGNOSTICS_SYSTEM_PROMPT = """You are the Kubernetes diagnostics worker agent.
You have access to the following tools to investigate Kubernetes clusters.

== Available Tools ==
- k8s_get_logs(namespace, service): Returns log data for pods behind a service label selector.
- k8s_get_metrics(namespace, duration): Collects Prometheus metrics and returns a directory path
  containing CSV files.
- k8s_read_metrics(file_path): Reads a metrics CSV file and returns its contents as text.
- k8s_get_traces(namespace, duration): Collects Jaeger traces and returns a directory path containing traces.csv.
- k8s_read_traces(file_path): Reads a traces CSV file and returns its contents as text.

You are also provided an API to a secure terminal to the service where you can run commands:

- k8s_exec_shell(command, timeout=30): Executes a shell command in the diagnostics environment and
  returns stdout/stderr (non-interactive, no stateful shell session).

Finally you need to answer the user with:

- A concise summary of your findings.
- Recommendations for next steps to resolve the user's goal.
- Structured JSON with the following fields:
- Start with “What I checked” (max 4 bullets).
- Then “Findings” with concrete evidence (key values/lines).
- Then “Conclusion: Healthy/Not healthy/Unclear” with confidence.
- Then “Next step” (one actionable step) only if needed.

Never attempt to mutate the cluster or guess. If information is missing, say so in the
summary and recommendations."""

_DIAGNOSTICS_AGENT = None
_DEFAULT_SIGNATURE_SECRET = "diagnostics-worker-signing-key"


def _post_mcp(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send a JSON-RPC request to the MCP server and return the decoded line."""
    response = _SESSION.post(url, headers=HDRS, json=payload, timeout=90, stream=True)
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            return json.loads(line[6:])
    raise RuntimeError("No 'data:' line from MCP")


def mcp_call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Invoke a named MCP tool and return its structured payload."""
    resp = _post_mcp(
        K8S_MCP_URL,
        {
            "jsonrpc": "2.0",
            "id": "supervisor",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    result = (resp or {}).get("result") or {}
    if result.get("structuredContent") is not None:
        return result["structuredContent"]
    for chunk in result.get("content") or []:
        if chunk.get("type") == "text":
            return chunk.get("text")
    return result


def _call_mcp_json(tool_name: str, **arguments: Any) -> str:
    """Convenience wrapper returning a JSON string to keep LangChain outputs uniform."""
    clean_args = {k: v for k, v in arguments.items() if v is not None}
    data = mcp_call_tool(tool_name, clean_args)
    # Keep compact to make responses easier to read in UI.
    return json.dumps(data, indent=2, ensure_ascii=False)


def _build_llm(model_spec: Optional[str] = None) -> Any:
    spec = model_spec or os.getenv("MODEL", "ollama:gpt-oss:20b-cloud")
    llm: Any = spec
    if spec.startswith("ollama:"):
        if ChatOllama is None:
            raise RuntimeError(
                "MODEL is set to an Ollama backend but langchain_ollama is not installed. "
                "Install with `pip install langchain-ollama`."
            )
        _, _, remaining = spec.partition(":")
        model_name = remaining or "ollama:gpt-oss:20b-cloud"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        llm = ChatOllama(model=model_name, base_url=base_url)
    return llm


def _build_tools(allowed_names: Optional[Iterable[str]] = None) -> Iterable[BaseTool]:
    """Construct LangChain tools that proxy to MCP endpoints."""
    allowed = set(allowed_names) if allowed_names is not None else None

    def wrap(fn, *, name: str, description: str) -> BaseTool:
        if StructuredTool is not None:
            return StructuredTool.from_function(fn, name=name, description=description)
        # Fallback to classic tool decorator which returns a Tool instance.
        decorated = tool(fn, return_direct=False)
        decorated.name = name
        decorated.description = description
        return decorated
    
    def exec_shell(command: str, timeout: int = 30) -> str:
        """Execute a shell command in the diagnostics environment."""
        return _call_mcp_json("exec_shell", command=command, timeout=timeout)


    def list_nodes() -> str:
        """List Kubernetes nodes and readiness state."""
        return _call_mcp_json("list_nodes")

    def list_namespaces() -> str:
        """List Kubernetes namespaces."""
        return _call_mcp_json("list_namespaces")

    def list_pods(namespace: str) -> str:
        """List pods in a namespace with phase/ready/restart details."""
        return _call_mcp_json("list_pods", namespace=namespace)

    def pod_events(namespace: str, pod: str) -> str:
        """List recent events for a pod."""
        return _call_mcp_json("pod_events", namespace=namespace, pod=pod)

    def pod_logs(namespace: str, pod: str, tail_lines: int = 80) -> str:
        """Get recent pod logs."""
        return _call_mcp_json("pod_logs", namespace=namespace, pod=pod, tail_lines=tail_lines)

    def get_logs(namespace: str, service: str) -> str:
        """Collect service logs for pods selected by the service."""
        return _call_mcp_json("get_logs", namespace=namespace, service=service)

    def get_metrics(namespace: str, duration: int = 5) -> str:
        """Collect Prometheus metrics and return the output directory."""
        return _call_mcp_json("get_metrics", namespace=namespace, duration=duration)

    def get_traces(namespace: str, duration: int = 5) -> str:
        """Collect Jaeger traces and return the output directory."""
        return _call_mcp_json("get_traces", namespace=namespace, duration=duration)

    def read_metrics(file_path: str) -> str:
        """Read metrics from a CSV file path."""
        return _call_mcp_json("read_metrics", file_path=file_path)

    def read_traces(file_path: str) -> str:
        """Read traces from a CSV file path."""
        return _call_mcp_json("read_traces", file_path=file_path)

    def create_deployment(namespace: str, name: str, image: str = "nginx", replicas: int = 1) -> str:
        """Create a small deployment."""
        return _call_mcp_json(
            "create_deployment",
            namespace=namespace,
            name=name,
            image=image,
            replicas=replicas,
        )

    def scale_deployment(namespace: str, name: str, replicas: int) -> str:
        """Scale an existing deployment."""
        return _call_mcp_json(
            "scale_deployment",
            namespace=namespace,
            name=name,
            replicas=replicas,
        )

    def delete_deployment(namespace: str, name: str, grace_period_seconds: Optional[int] = None) -> str:
        """Delete a deployment after confirming namespace/name."""
        return _call_mcp_json(
            "delete_deployment",
            namespace=namespace,
            name=name,
            grace_period_seconds=grace_period_seconds,
        )

    def delete_pod(namespace: str, name: str, grace_period_seconds: Optional[int] = None) -> str:
        """Delete a pod and scale the owning deployment down when applicable."""
        return _call_mcp_json(
            "delete_pod",
            namespace=namespace,
            name=name,
            grace_period_seconds=grace_period_seconds,
        )
    

    def run_diagnostics(
        goal: str,
        namespace: Optional[str] = None,
        workload: Optional[str] = None,
        include_logs: bool = True,
        max_pods: int = 3,
    ) -> str:
        """Delegate to the diagnostics worker for a scoped investigation."""
        return _run_diagnostics_worker(
            goal=goal,
            namespace=namespace,
            workload=workload,
            include_logs=include_logs,
            max_pods=max_pods,
        )

    tool_specs = [
        ("k8s_exec_shell", exec_shell, exec_shell.__doc__ or ""),
        ("k8s_list_nodes", list_nodes, list_nodes.__doc__ or ""),
        ("k8s_list_namespaces", list_namespaces, list_namespaces.__doc__ or ""),
        ("k8s_list_pods", list_pods, list_pods.__doc__ or ""),
        ("k8s_pod_events", pod_events, pod_events.__doc__ or ""),
        ("k8s_pod_logs", pod_logs, pod_logs.__doc__ or ""),
        ("k8s_get_logs", get_logs, get_logs.__doc__ or ""),
        ("k8s_get_metrics", get_metrics, get_metrics.__doc__ or ""),
        ("k8s_get_traces", get_traces, get_traces.__doc__ or ""),
        ("k8s_read_metrics", read_metrics, read_metrics.__doc__ or ""),
        ("k8s_read_traces", read_traces, read_traces.__doc__ or ""),
        ("k8s_create_deployment", create_deployment, create_deployment.__doc__ or ""),
        ("k8s_scale_deployment", scale_deployment, scale_deployment.__doc__ or ""),
        ("k8s_delete_deployment", delete_deployment, delete_deployment.__doc__ or ""),
        ("k8s_delete_pod", delete_pod, delete_pod.__doc__ or ""),
        ("k8s_run_diagnostics", run_diagnostics, run_diagnostics.__doc__ or ""),
    ]

    tools: List[BaseTool] = []
    for name, fn, description in tool_specs:
       
        if allowed is not None and name not in allowed:
            continue

        
        if allowed is None and name in SUPERVISOR_TOOL_DENYLIST:
            continue

        tools.append(wrap(fn, name=name, description=description))
    return tools


def _extract_answer(payload: Dict[str, Any]) -> str:
    answer = payload.get("output")
    if answer:
        return answer
    messages = payload.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    chunk.get("text")
                    for chunk in content
                    if isinstance(chunk, dict) and chunk.get("type") == "text"
                ]
                if parts:
                    return "\n".join(filter(None, parts))
        if isinstance(message, dict) and message.get("role") == "assistant":
            assistant_content = message.get("content")
            if isinstance(assistant_content, str):
                return assistant_content
    raise RuntimeError("Diagnostics worker returned no output.")


def _get_diagnostics_agent():
    global _DIAGNOSTICS_AGENT
    if _DIAGNOSTICS_AGENT is None:
        diag_model = os.getenv("DIAGNOSTICS_MODEL", "ollama:gpt-oss:20b-cloud")
        tools = list(_build_tools(allowed_names=DIAGNOSTICS_TOOL_ALLOWLIST))
        _DIAGNOSTICS_AGENT = create_agent(
            model=_build_llm(diag_model),
            tools=tools,
            system_prompt=DIAGNOSTICS_SYSTEM_PROMPT,
            checkpointer=DIAGNOSTICS_CHECKPOINTER,
        )
    return _DIAGNOSTICS_AGENT


def _run_diagnostics_worker(
    goal: str,
    namespace: Optional[str],
    workload: Optional[str],
    include_logs: bool,
    max_pods: int,
) -> str:
    clean_goal = (goal or "").strip()
    if not clean_goal:
        raise ValueError("goal is required for diagnostics")
    if max_pods <= 0:
        raise ValueError("max_pods must be a positive integer")

    worker = _get_diagnostics_agent()
    focus_namespace = namespace.strip() if namespace else "all namespaces"
    focus_workload = workload.strip() if workload else "any workload"
    lines = [
        f"Goal: {clean_goal}",
        f"Namespace focus: {focus_namespace}",
        f"Workload focus: {focus_workload}",
        f"Collect logs: {'yes' if include_logs else 'no'}",
        f"Max pods for deep inspection: {max_pods}",
        "Follow the diagnostics workflow and return the summary plus structured JSON.",
    ]
    config = {"configurable": {"thread_id": f"diag-{uuid4()}"}}
    result = worker.invoke({"messages": [HumanMessage(content="\n".join(lines))]}, config=config)
    logger.info("[DIAG-AGENT] diagnose_cluster() called")
    return _attach_worker_signatures(_extract_answer(result))



def build_agent_v1(system_prompt: str) -> Any:
    """Create the LangChain agent wired up with MCP-backed tools."""
    tools = list(_build_tools())
    llm = _build_llm()
    logger.info("Supervisor LLM type: %r", llm)
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on=HITL_POLICY,
                description_prefix="Tool execution pending approval",
            )
        ],
        checkpointer=CHECKPOINTER,
    )

def _attach_worker_signatures(answer: str) -> str:
    """Append deterministic signatures so responses can be verified."""
    payload = answer or ""
    secret = os.getenv("WORKER_SIGNATURE_SECRET", _DEFAULT_SIGNATURE_SECRET).encode("utf-8")
    body = payload.encode("utf-8")
    hmac_sha256 = hmac.new(secret, body, hashlib.sha256).hexdigest()
    blake2s_sig = hashlib.blake2s(body + secret).hexdigest()
    signature_block = (
        "\n\n---\n"
        "Worker agent signatures:\n"
        f"- hmac_sha256: {hmac_sha256}\n"
        f"- blake2s: {blake2s_sig}\n"
    )
    return payload.rstrip("\n") + signature_block
