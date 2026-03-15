import unittest

from langchain.messages import HumanMessage

from supervisor.agents import SupervisorWorkflow, classify_intent


class FakeOperationsAgent:
    def __init__(self):
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append(payload)
        text = payload["messages"][0].content
        return {"output": f"operations:{text}"}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.ops = FakeOperationsAgent()
        self.trace = []

        def list_pods(namespace="default"):
            self.trace.append("detection")
            return {
                "pods": [
                    {"name": "api-0", "phase": "Running"},
                    {"name": "api-1", "phase": "CrashLoopBackOff"},
                ]
            }

        def pod_events(namespace, pod):
            self.trace.append("localization")
            return {"events": [{"reason": "BackOff"}]}

        def pod_logs(namespace, pod, tail_lines=80):
            return {"logs": "crash"}

        def exec_shell(command, timeout=30):
            self.trace.append("analysis")
            return {"stdout": "kubectl output"}

        self.workflow = SupervisorWorkflow(
            operations_agent=self.ops,
            diagnostics_tools={
                "list_pods": list_pods,
                "pod_events": pod_events,
                "pod_logs": pod_logs,
                "exec_shell": exec_shell,
            },
        )

    def test_router_classification(self):
        self.assertEqual(classify_intent("list namespaces"), "inspect")
        self.assertEqual(classify_intent("scale deployment api to 3"), "act")
        self.assertEqual(classify_intent("diagnose my cluster"), "diagnose")

    def test_inspect_bypasses_diagnostics(self):
        state = {"messages": [HumanMessage(content="list pods in default")], "user_request": "list pods in default"}
        routed = self.workflow.router_node(state)
        self.assertEqual(routed["intent"], "inspect")
        result = self.workflow.operations_node({"user_request": "list pods in default"})
        self.assertIn("operations:", result["final_response"])
        self.assertEqual(self.trace, [])

    def test_act_bypasses_diagnostics(self):
        routed = self.workflow.router_node({"user_request": "delete pod api-1"})
        self.assertEqual(routed["intent"], "act")
        result = self.workflow.operations_node({"user_request": "delete pod api-1"})
        self.assertIn("delete pod", result["final_response"])
        self.assertEqual(self.trace, [])

    def test_diagnose_runs_detection_localization_analysis(self):
        state = {"user_request": "diagnose my cluster"}
        state.update(self.workflow.router_node(state))
        state.update(self.workflow.detection_node(state))
        self.assertTrue(state["issue_found"])
        state.update(self.workflow.localization_node(state))
        state.update(self.workflow.analysis_node(state))
        state.update(self.workflow.mitigation_node(state))
        self.assertEqual(self.trace, ["detection", "localization", "analysis"])
        self.assertIn("Proposed mitigation", state["final_response"])

    def test_detection_handles_list_pods_tool_shape_and_restarts(self):
        def list_pods_as_list(namespace="default"):
            self.trace.append("detection")
            return [
                {
                    "name": "api-0",
                    "phase": "Running",
                    "restarts": 2,
                    "containers": [{"state": {"state": "running"}}],
                }
            ]

        workflow = SupervisorWorkflow(
            operations_agent=self.ops,
            diagnostics_tools={
                "list_pods": list_pods_as_list,
                "pod_events": lambda namespace, pod: {},
                "pod_logs": lambda namespace, pod, tail_lines=80: {},
                "exec_shell": lambda command, timeout=30: {},
            },
        )
        result = workflow.detection_node({"user_request": "diagnose namespace default"})
        self.assertTrue(result["issue_found"])
        self.assertEqual(result["diagnostics_summary"]["detection"]["unhealthy_pods"], ["api-0"])

    def test_mitigation_execution_goes_through_operations_agent(self):
        state = {
            "user_request": "diagnose and execute mitigation",
            "location": "default/api-1",
            "issue_found": True,
            "analysis_result": "bad pod",
        }
        state.update(self.workflow.mitigation_node(state))
        self.assertTrue(state["execute_mitigation"])
        result = self.workflow.mitigation_to_operations_node(state)
        self.assertIn("Mitigation execution result", result["final_response"])
        self.assertEqual(len(self.ops.calls), 1)


if __name__ == "__main__":
    unittest.main()
