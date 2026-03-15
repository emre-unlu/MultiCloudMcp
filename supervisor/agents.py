import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, TypedDict

import requests
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

try:
    from langchain.tools import StructuredTool
except ImportError:  # pragma: no cover
    StructuredTool = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

K8S_MCP_URL = os.getenv("K8S_MCP_URL", "http://127.0.0.1:8080")
HDRS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
_SESSION = requests.Session()
CHECKPOINTER = InMemorySaver()

INSPECT_INTENT_KEYWORDS = {
    "list",
    "show",
    "inspect",
    "describe",
    "status",
    "get",
    "namespace",
    "pod",
    "deployment",
}
ACT_INTENT_KEYWORDS = {
    "create",
    "delete",
    "scale",
    "restart",
    "patch",
    "apply",
    "rollout",
}
DIAGNOSE_INTENT_KEYWORDS = {
    "diagnose",
    "diagnostics",
    "health",
    "issue",
    "problem",
    "root cause",
    "analyze",
    "mitigation",
    "failing",
    "not working",
    "anomaly",
}


class SupervisorState(TypedDict, total=False):
    user_request: str
    intent: Literal["inspect", "act", "diagnose"]
    issue_found: bool
    location: str
    analysis_result: str
    mitigation_plan: Dict[str, Any]
    final_response: str
    operation_result: Dict[str, Any]
    execute_mitigation: bool
    diagnostics_summary: Dict[str, Any]


def _post_mcp(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = _SESSION.post(url, headers=HDRS, json=payload, timeout=90, stream=True)
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            return json.loads(line[6:])
    raise RuntimeError("No 'data:' line from MCP")


def mcp_call_tool(name: str, arguments: Dict[str, Any]) -> Any:
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
    clean_args = {k: v for k, v in arguments.items() if v is not None}
    data = mcp_call_tool(tool_name, clean_args)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _extract_answer(payload: Dict[str, Any]) -> str:
    answer = payload.get("output")
    if answer:
        return answer
    messages = payload.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def _build_llm(model_spec: Optional[str] = None) -> Any:
    model_name = model_spec or os.getenv("MODEL", "gpt-5-mini-2025-08-07")
    if ChatOpenAI is None:
        raise RuntimeError("Install langchain-openai to use OpenAI models.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please export it before starting the supervisor.")
    return ChatOpenAI(model=model_name, api_key=api_key)


def classify_intent(user_request: str) -> Literal["inspect", "act", "diagnose"]:
    text = (user_request or "").lower()
    if any(keyword in text for keyword in DIAGNOSE_INTENT_KEYWORDS):
        return "diagnose"
    if any(keyword in text for keyword in ACT_INTENT_KEYWORDS):
        return "act"
    if any(keyword in text for keyword in INSPECT_INTENT_KEYWORDS):
        return "inspect"
    return "inspect"


def _build_operations_tools() -> Iterable[BaseTool]:
    def wrap(fn, *, name: str, description: str) -> BaseTool:
        if StructuredTool is not None:
            return StructuredTool.from_function(fn, name=name, description=description)
        decorated = tool(fn, return_direct=False)
        decorated.name = name
        decorated.description = description
        return decorated

    def list_namespaces() -> str:
        """List Kubernetes namespaces."""
        return _call_mcp_json("list_namespaces")

    def list_pods(namespace: str = "default") -> str:
        """List pods for a namespace."""
        return _call_mcp_json("list_pods", namespace=namespace)

    def describe_resource(kind: str, name: str, namespace: str = "default") -> str:
        """Describe a Kubernetes resource."""
        command = f"kubectl describe {kind} {name} -n {namespace}"
        return _call_mcp_json("exec_shell", command=command, timeout=30)

    def create_pod(namespace: str, name: str, image: str) -> str:
        """Create a pod from an image."""
        command = f"kubectl run {name} --image={image} -n {namespace}"
        return _call_mcp_json("exec_shell", command=command, timeout=30)

    def delete_pod(namespace: str, name: str, grace_period_seconds: Optional[int] = None) -> str:
        """Delete a pod by name."""
        return _call_mcp_json("delete_pod", namespace=namespace, name=name, grace_period_seconds=grace_period_seconds)

    def scale_deployment(namespace: str, name: str, replicas: int) -> str:
        """Scale a deployment to desired replicas."""
        return _call_mcp_json("scale_deployment", namespace=namespace, name=name, replicas=replicas)

    def restart_rollout(namespace: str, name: str) -> str:
        """Restart deployment rollout."""
        command = f"kubectl rollout restart deployment/{name} -n {namespace}"
        return _call_mcp_json("exec_shell", command=command, timeout=30)

    def patch_resource(kind: str, name: str, patch: str, namespace: str = "default") -> str:
        """Patch a Kubernetes resource with JSON patch string."""
        command = f"kubectl patch {kind} {name} -n {namespace} -p '{patch}'"
        return _call_mcp_json("exec_shell", command=command, timeout=30)

    specs = [
        ("list_namespaces", list_namespaces),
        ("list_pods", list_pods),
        ("describe_resource", describe_resource),
        ("create_pod", create_pod),
        ("delete_pod", delete_pod),
        ("scale_deployment", scale_deployment),
        ("restart_rollout", restart_rollout),
        ("patch_resource", patch_resource),
    ]
    return [wrap(fn, name=name, description=fn.__doc__ or "") for name, fn in specs]


def _build_diagnostics_tools() -> Dict[str, Callable[..., Any]]:
    return {
        "get_logs": lambda namespace, service: mcp_call_tool("get_logs", {"namespace": namespace, "service": service}),
        "get_metrics": lambda namespace, duration=5: mcp_call_tool("get_metrics", {"namespace": namespace, "duration": duration}),
        "get_traces": lambda namespace, duration=5: mcp_call_tool("get_traces", {"namespace": namespace, "duration": duration}),
        "read_metrics": lambda file_path: mcp_call_tool("read_metrics", {"file_path": file_path}),
        "read_traces": lambda file_path: mcp_call_tool("read_traces", {"file_path": file_path}),
        "pod_events": lambda namespace, pod: mcp_call_tool("pod_events", {"namespace": namespace, "pod": pod}),
        "pod_logs": lambda namespace, pod, tail_lines=80: mcp_call_tool(
            "pod_logs", {"namespace": namespace, "pod": pod, "tail_lines": tail_lines}
        ),
        "list_pods": lambda namespace="default": mcp_call_tool("list_pods", {"namespace": namespace}),
        "exec_shell": lambda command, timeout=30: mcp_call_tool("exec_shell", {"command": command, "timeout": timeout}),
    }


@dataclass
class SupervisorWorkflow:
    operations_agent: Any
    diagnostics_tools: Dict[str, Callable[..., Any]]

    def router_node(self, state: SupervisorState) -> SupervisorState:
        return {"intent": classify_intent(state.get("user_request", ""))}

    def operations_node(self, state: SupervisorState, config: RunnableConfig | None = None) -> SupervisorState:
        request = state.get("user_request", "")
        result = self.operations_agent.invoke({"messages": [HumanMessage(content=request)]}, config=config)
        update: SupervisorState = {"operation_result": result}
        if result.get("__interrupt__"):
            update["final_response"] = "Approval required before executing operations action."
            return update
        update["final_response"] = _extract_answer(result)
        return update

    def detection_node(self, state: SupervisorState) -> SupervisorState:
        text = state.get("user_request", "")
        namespace_match = re.search(r"namespace\s+([a-z0-9-]+)", text.lower())
        namespace = namespace_match.group(1) if namespace_match else "default"
        pods = self.diagnostics_tools["list_pods"](namespace)
        pod_items = pods.get("pods") if isinstance(pods, dict) else None
        issue_found = False
        unhealthy: List[str] = []
        if isinstance(pod_items, list):
            for pod in pod_items:
                phase = str(pod.get("phase", "")).lower()
                if phase not in {"running", "succeeded"}:
                    unhealthy.append(pod.get("name", "unknown"))
            issue_found = bool(unhealthy)
        summary = {
            "namespace": namespace,
            "pods_checked": len(pod_items or []),
            "unhealthy_pods": unhealthy,
        }
        return {
            "issue_found": issue_found,
            "location": namespace,
            "diagnostics_summary": {"detection": summary},
            "final_response": "No issue detected." if not issue_found else state.get("final_response", ""),
        }

    def localization_node(self, state: SupervisorState) -> SupervisorState:
        if not state.get("issue_found"):
            return {}
        namespace = state.get("location", "default")
        detection = (state.get("diagnostics_summary") or {}).get("detection", {})
        suspect = (detection.get("unhealthy_pods") or ["unknown"])[0]
        events = self.diagnostics_tools["pod_events"](namespace, suspect)
        logs = self.diagnostics_tools["pod_logs"](namespace, suspect, 80)
        summary = {"suspect_pod": suspect, "events": events, "logs": logs}
        merged = dict(state.get("diagnostics_summary") or {})
        merged["localization"] = summary
        return {"diagnostics_summary": merged, "location": f"{namespace}/{suspect}"}

    def analysis_node(self, state: SupervisorState) -> SupervisorState:
        if not state.get("issue_found"):
            return {}
        namespace = state.get("location", "default").split("/")[0]
        shell = self.diagnostics_tools["exec_shell"](f"kubectl get pods -n {namespace}")
        merged = dict(state.get("diagnostics_summary") or {})
        merged["analysis"] = {"kubectl_get_pods": shell}
        analysis_text = f"Potential issue localized to {state.get('location')}."
        return {"diagnostics_summary": merged, "analysis_result": analysis_text}

    def mitigation_node(self, state: SupervisorState) -> SupervisorState:
        if not state.get("issue_found"):
            return {"mitigation_plan": {}, "final_response": "No issue detected during diagnostics."}
        location = state.get("location", "default/unknown")
        namespace, _, pod = location.partition("/")
        plan = {
            "summary": f"Investigate and recover pod {pod} in namespace {namespace}.",
            "actions": [
                f"describe_resource pod {pod} -n {namespace}",
                f"delete_pod {namespace} {pod}",
            ],
        }
        request_text = state.get("user_request", "").lower()
        execute = any(token in request_text for token in ["execute", "apply", "run mitigation", "approve mitigation"])
        response = (
            f"Detection found issues. Localization: {location}. Analysis: {state.get('analysis_result', '')}\n"
            f"Proposed mitigation: {plan['summary']}\nActions: {', '.join(plan['actions'])}"
        )
        return {"mitigation_plan": plan, "execute_mitigation": execute, "final_response": response}

    def mitigation_to_operations_node(self, state: SupervisorState, config: RunnableConfig | None = None) -> SupervisorState:
        plan = state.get("mitigation_plan") or {}
        namespace = state.get("location", "default/unknown").split("/")[0]
        pod = state.get("location", "default/unknown").split("/")[-1]
        mitigation_request = (
            f"Execute approved mitigation now. "
            f"Use describe_resource for pod {pod} in namespace {namespace}, then delete_pod namespace={namespace} name={pod}."
        )
        op_state: SupervisorState = {"user_request": mitigation_request}
        result = self.operations_node(op_state, config=config)
        final = f"{state.get('final_response', '')}\n\nMitigation execution result:\n{result.get('final_response', '')}"
        return {"final_response": final, "operation_result": result.get("operation_result", {})}


def _build_operations_agent() -> Any:
    hitl_policy = {
        "create_pod": True,
        "delete_pod": True,
        "scale_deployment": True,
        "restart_rollout": True,
        "patch_resource": True,
        "list_namespaces": False,
        "list_pods": False,
        "describe_resource": False,
    }
    return create_agent(
        model=_build_llm(),
        tools=list(_build_operations_tools()),
        system_prompt=(
            "You are the Kubernetes operations agent. Handle inspect and act requests with provided tools. "
            "For act requests perform only requested actions; for mitigation execution follow explicit instructions."
        ),
        middleware=[HumanInTheLoopMiddleware(interrupt_on=hitl_policy, description_prefix="Tool execution pending approval")],
        checkpointer=CHECKPOINTER,
    )


def build_agent_v1(system_prompt: str) -> Any:  # system_prompt kept for compatibility
    workflow = SupervisorWorkflow(
        operations_agent=_build_operations_agent(),
        diagnostics_tools=_build_diagnostics_tools(),
    )

    graph = StateGraph(SupervisorState)

    def bootstrap_node(state: Dict[str, Any]) -> SupervisorState:
        messages = state.get("messages") or []
        user_request = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_request = message.content
                break
            if isinstance(message, dict) and message.get("role") == "user":
                user_request = str(message.get("content", ""))
                break
        if not user_request and state.get("user_request"):
            user_request = str(state["user_request"])
        return {"user_request": user_request}

    graph.add_node("bootstrap", bootstrap_node)
    graph.add_node("router", workflow.router_node)
    graph.add_node("operations", workflow.operations_node)
    graph.add_node("detection", workflow.detection_node)
    graph.add_node("localization", workflow.localization_node)
    graph.add_node("analysis", workflow.analysis_node)
    graph.add_node("mitigation", workflow.mitigation_node)
    graph.add_node("mitigation_execute", workflow.mitigation_to_operations_node)

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "router")

    def route_from_router(state: SupervisorState) -> str:
        return "detection" if state.get("intent") == "diagnose" else "operations"

    graph.add_conditional_edges("router", route_from_router, {"operations": "operations", "detection": "detection"})

    def detection_gate(state: SupervisorState) -> str:
        return "end" if not state.get("issue_found") else "localization"

    graph.add_conditional_edges("detection", detection_gate, {"end": END, "localization": "localization"})
    graph.add_edge("localization", "analysis")
    graph.add_edge("analysis", "mitigation")

    def mitigation_gate(state: SupervisorState) -> str:
        return "mitigation_execute" if state.get("execute_mitigation") else "end"

    graph.add_conditional_edges("mitigation", mitigation_gate, {"mitigation_execute": "mitigation_execute", "end": END})
    graph.add_edge("mitigation_execute", END)
    graph.add_edge("operations", END)

    return graph.compile(checkpointer=CHECKPOINTER)
