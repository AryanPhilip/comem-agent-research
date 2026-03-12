import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.getcwd(), "CoMEM-Agent-Inference"))

from agent.controller import StepVerifier, build_structured_page_state  # noqa: E402


class ControllerTest(unittest.TestCase):
    def test_verifier_detects_repeated_action_loop(self):
        verifier = StepVerifier()
        state = build_structured_page_state(
            current_url="https://example.com/search",
            action_history=["click result", "click result", "click result"],
            current_subgoal="find the result",
            failure_state="",
        )

        result = verifier.verify(
            intent="find the result",
            state=state,
            action_history=["click result", "click result", "click result"],
            response_history=[],
            site="example.com",
        )

        self.assertTrue(result.needs_refresh)
        self.assertEqual(result.failure_state, "repeated_action_loop")

    def test_verifier_detects_page_drift(self):
        verifier = StepVerifier()
        state = build_structured_page_state(
            current_url="https://www.google.com/search?q=github",
            action_history=["type query"],
            current_subgoal="open github issue",
            failure_state="",
        )

        result = verifier.verify(
            intent="open github issue",
            state=state,
            action_history=["type query"],
            response_history=[],
            site="github.com",
        )

        self.assertTrue(result.needs_refresh)
        self.assertEqual(result.failure_state, "wrong_page")


if __name__ == "__main__":
    unittest.main()
