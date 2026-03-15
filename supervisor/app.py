"""FastAPI surface for the Supervisor agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Set, Tuple

from fastapi import FastAPI, HTTPException
from langchain.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field
from supervisor.prompts import SYSTEM_PROMPT

from .agents import build_agent_v1

app = FastAPI()
_agent = None
DEFAULT_THREAD_ID = "ui-thread"
PENDING_INTERRUPTS: Dict[str, Dict[str, Any]] = {}


class RunReq(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str | None = Field(default=None)


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent_v1(SYSTEM_PROMPT)
    return _agent


def _flatten_interrupts(interrupts: List[Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    requests: List[Dict[str, Any]] = []
    reviews: List[Dict[str, Any]] = []
    for interrupt in interrupts:
        data = getattr(interrupt, "value", interrupt)
        requests.extend(data.get("action_requests") or [])
        reviews.extend(data.get("review_configs") or [])
    return requests, reviews


def _allowed_decisions(review_configs: List[Dict[str, Any]]) -> Set[str]:
    allowed_sets: List[Set[str]] = []
    for config in review_configs:
        decisions = config.get("allowed_decisions")
        if decisions:
            allowed_sets.append(set(decisions))
    if not allowed_sets:
        return {"approve", "edit", "reject"}
    allowed = allowed_sets[0].copy()
    for other in allowed_sets[1:]:
        allowed &= other
    return allowed or allowed_sets[0]


def _build_interrupt_prompt(requests: List[Dict[str, Any]], allowed: Set[str]) -> str:
    lines = ["Approval required before executing the following tool calls:"]
    for idx, request in enumerate(requests, start=1):
        name = request.get("name", "<unknown>")
        args = request.get("arguments") or request.get("args") or {}
        lines.append(f"{idx}. {name}")
        if args:
            lines.append(json.dumps(args, indent=2))
    allowed_text = ", ".join(sorted(allowed))
    lines.append("")
    lines.append(f"Reply with one of: {allowed_text}.")
    lines.append("Examples: 'approve', 'reject this change'.")
    return "\n".join(lines)


def _parse_decision(message: str, allowed: Set[str]) -> Tuple[str, str | None]:
    text = (message or "").strip()
    if not text:
        return ("invalid", "Please reply with a decision (approve or reject).")
    lower = text.lower()
    if "approve" in lower or lower in {"yes", "y", "ok", "okay", "go ahead", "proceed"}:
        if "approve" not in allowed:
            return ("invalid", "Approval is not permitted for this action.")
        return ("approve", None)
    if lower.startswith("reject") or lower.startswith("no") or lower.startswith("deny") or "reject" in lower:
        if "reject" not in allowed:
            return ("invalid", "Rejection is not permitted for this action.")
        reason = text.partition(" ")[2].strip()
        if not reason:
            reason = "Rejected by reviewer."
        return ("reject", reason)
    if lower.startswith("edit"):
        if "edit" not in allowed:
            return ("invalid", "Editing is not permitted for this action.")
        return ("unsupported_edit", None)
    return ("invalid", "Please reply with 'approve' or 'reject <reason>'.")


def _normalize_answer_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content)


def _register_interrupt(thread_id: str, interrupts: List[Any]) -> Tuple[str, Set[str]]:
    requests, reviews = _flatten_interrupts(interrupts)
    allowed = _allowed_decisions(reviews)
    prompt = _build_interrupt_prompt(requests, allowed)
    PENDING_INTERRUPTS[thread_id] = {
        "interrupts": interrupts,
        "requests": requests,
        "reviews": reviews,
        "allowed": allowed,
        "prompt": prompt,
    }
    return prompt, allowed


@app.get("/health")
def health():
    try:
        get_agent()
    except Exception as exc:  # pragma: no cover - surfaced via API
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


@app.post("/run")
def run(req: RunReq):
    agent = get_agent()
    content = (req.message or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message is required.")

    if req.thread_id is not None:
        tid = req.thread_id.strip()
        if not tid:
            raise HTTPException(status_code=400, detail="thread_id cannot be blank.")
        thread_id = tid
    else:
        thread_id = DEFAULT_THREAD_ID

    config = {"configurable": {"thread_id": thread_id}}

    if thread_id in PENDING_INTERRUPTS:
        pending = PENDING_INTERRUPTS[thread_id]
        decision, detail = _parse_decision(content, pending["allowed"])
        if decision == "invalid":
            answer = f"{pending['prompt']}\n\n{detail}"
            return {
                "ok": True,
                "answer": answer,
                "thread_id": thread_id,
                "awaiting_decision": True,
            }
        if decision == "unsupported_edit":
            answer = (
                "Editing tool arguments is not supported in this interface. "
                "Please reply with 'approve' or 'reject <reason>'."
            )
            return {
                "ok": True,
                "answer": answer,
                "thread_id": thread_id,
                "awaiting_decision": True,
            }

        if decision == "approve":
            decisions_payload = [{"type": "approve"} for _ in pending["requests"]]
        elif decision == "reject":
            message_text = detail or "Rejected by reviewer."
            decisions_payload = [
                {"type": "reject", "message": message_text}
                for _ in pending["requests"]
            ]
        else:
            decisions_payload = [{"type": decision}]

        result = agent.invoke(
            Command(resume={"decisions": decisions_payload}),
            config=config,
        )
        PENDING_INTERRUPTS.pop(thread_id, None)
    else:
        result = agent.invoke({"messages": [HumanMessage(content=content)]}, config=config)

    answer = _normalize_answer_content(result.get("final_response"))
    if not answer:
        answer = _normalize_answer_content(result.get("output"))
    if not answer:
        messages = result.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                answer = _normalize_answer_content(message.content)
                break
            if isinstance(message, dict) and message.get("role") == "assistant":
                answer = _normalize_answer_content(message.get("content"))
                break

    interrupts = result.get("__interrupt__")
    if interrupts:
        prompt, allowed = _register_interrupt(thread_id, interrupts)
        requests = PENDING_INTERRUPTS[thread_id]["requests"]
        return {
            "ok": True,
            "answer": prompt,
            "thread_id": thread_id,
            "awaiting_decision": True,
            "pending_actions": [
                {"name": req.get("name"), "arguments": req.get("arguments") or req.get("args") or {}}
                for req in requests
            ],
            "allowed_decisions": sorted(allowed),
        }

    return {"ok": True, "answer": answer or "(no final answer)", "thread_id": thread_id}
