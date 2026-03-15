import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from supervisor import app as app_module


class FakeAgent:
    def __init__(self, result):
        self.result = result

    def invoke(self, payload, config=None):
        return self.result


class EchoUserRequestAgent:
    def invoke(self, payload, config=None):
        return {"final_response": payload.get("user_request", "")}


class RunEndpointTests(unittest.TestCase):
    def setUp(self):
        app_module.PENDING_INTERRUPTS.clear()
        self.client = TestClient(app_module.app)

    def test_run_prefers_final_response_when_present(self):
        fake_agent = FakeAgent({"final_response": "done from final response"})

        with patch.object(app_module, "get_agent", return_value=fake_agent):
            response = self.client.post("/run", json={"message": "hello"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["answer"], "done from final response")


    def test_run_passes_user_request_alongside_messages(self):
        echo_agent = EchoUserRequestAgent()

        with patch.object(app_module, "get_agent", return_value=echo_agent):
            response = self.client.post("/run", json={"message": "list pods in default"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "list pods in default")

    def test_run_normalizes_non_string_assistant_message_content(self):
        fake_agent = FakeAgent(
            {
                "messages": [
                    {"role": "assistant", "content": {"status": "ok", "count": 2}},
                ]
            }
        )

        with patch.object(app_module, "get_agent", return_value=fake_agent):
            response = self.client.post("/run", json={"message": "hello"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], '{"status": "ok", "count": 2}')


if __name__ == "__main__":
    unittest.main()
