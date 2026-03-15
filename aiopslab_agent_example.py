import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from aiopslab.orchestrator import Orchestrator
from langchain.messages import AIMessage, HumanMessage

from supervisor.agents import build_agent_v1


def _extract_answer(payload: dict) -> str:
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
    return ""


@dataclass
class Agent:

    def setup_safe_context(self, problem_desc, instructions, apis):

        system_prompt = (
            f"{problem_desc}\n\n"
            f"{instructions}\n\n"
            f"=== AVAILABLE APIs ===\n{apis}\n"
        )

        self.history = [{"role": "system", "content": system_prompt}]


    """AIOpsLab agent wrapper that uses the Supervisor + Diagnostics logic."""

    thread_id: str = field(default_factory=lambda: f"aiopslab-{uuid4()}")
    agent: object = field(default_factory=lambda: build_agent_v1(SYSTEM_PROMPT))

    async def get_action(self, state: str) -> str:
        config = {"configurable": {"thread_id": self.thread_id}}
        payload = await asyncio.to_thread(
            self.agent.invoke,
            {"messages": [HumanMessage(content=state)]},
            config,
        )
        return _extract_answer(payload)


async def main() -> None:
    agent = Agent()
    orch = Orchestrator()
    orch.register_agent(agent)

    problem_id = "k8s_target_port-misconfig-mitigation-1"
    problem_desc, instructs, apis = orch.init_problem(problem_id)

    # You can set context for your agent here using problem_desc/instructs/apis.
    # For example, log them or store them on the agent instance.
    SYSTEM_PROMPT = (problem_desc, instructs, apis)
    agent.setup_safe_context(problem_desc, instructs, apis)

    await orch.start_problem(max_steps=30)


if __name__ == "__main__":
    asyncio.run(main())

