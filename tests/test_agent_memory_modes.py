import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.getcwd(), "CoMEM-Agent-Inference"))

from memory.runtime import MemoryBundle, build_memory_runtime_payload  # noqa: E402


class AgentMemoryModesTest(unittest.TestCase):
    def setUp(self):
        self.bundle = MemoryBundle(
            prompt_text="EXAMPLE: choose the best result",
            experience_texts=[
                [{"name": "click", "arguments": {"reasoning": "open result", "description": ""}}]
            ],
            experience_images=[[""]],
            file_id_list=["memory-demo"],
            semantic_notes=["Workflow: search -> open result."],
        )

    def test_none_mode_uses_fallback_without_continuous_payload(self):
        payload = build_memory_runtime_payload(
            memory_mode="none",
            bundle=self.bundle,
            fallback_prompt_text="fallback example",
            memory_token_budget=8,
        )
        self.assertEqual(payload["prompt_text"], "fallback example")
        self.assertEqual(payload["chat_kwargs"], {})

    def test_text_mode_uses_prompt_only(self):
        payload = build_memory_runtime_payload(
            memory_mode="text",
            bundle=self.bundle,
            fallback_prompt_text="fallback example",
            memory_token_budget=8,
        )
        self.assertIn("EXAMPLE", payload["prompt_text"])
        self.assertEqual(payload["chat_kwargs"], {})

    def test_continuous_mode_uses_semantic_prompt_and_payload(self):
        payload = build_memory_runtime_payload(
            memory_mode="continuous",
            bundle=self.bundle,
            fallback_prompt_text="fallback example",
            memory_token_budget=8,
        )
        self.assertIn("Workflow", payload["prompt_text"])
        self.assertIn("experience_texts", payload["chat_kwargs"])
        self.assertEqual(payload["chat_kwargs"]["memory_token_budget"], 8)

    def test_hybrid_mode_uses_both_prompt_and_continuous_payload(self):
        payload = build_memory_runtime_payload(
            memory_mode="hybrid",
            bundle=self.bundle,
            fallback_prompt_text="fallback example",
            memory_token_budget=8,
            reflection_note="avoid repeated loops",
        )
        self.assertIn("EXAMPLE", payload["prompt_text"])
        self.assertIn("CORRECTIVE REFLECTION", payload["prompt_text"])
        self.assertIn("experience_texts", payload["chat_kwargs"])


if __name__ == "__main__":
    unittest.main()
