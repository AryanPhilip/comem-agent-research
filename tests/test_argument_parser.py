import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.getcwd(), "CoMEM-Agent-Inference"))

from config.argument_parser import _normalize_memory_args  # noqa: E402


class ArgumentParserNormalizationTest(unittest.TestCase):
    def test_legacy_flags_map_to_hybrid_mode(self):
        args = argparse.Namespace(
            memory_mode=None,
            use_memory=True,
            use_continuous_memory=True,
            use_text_memory=False,
            memory_token_budget=0,
            model="qwen2.5-vl",
        )

        normalized = _normalize_memory_args(args)
        self.assertEqual(normalized.memory_mode, "hybrid")
        self.assertTrue(normalized.use_text_memory)
        self.assertTrue(normalized.use_continuous_memory)
        self.assertEqual(normalized.model, "agent-qformer")
        self.assertEqual(normalized.memory_token_budget, 8)

    def test_explicit_none_mode_disables_memory(self):
        args = argparse.Namespace(
            memory_mode="none",
            use_memory=False,
            use_continuous_memory=False,
            use_text_memory=False,
            memory_token_budget=8,
            model="qwen2.5-vl",
        )

        normalized = _normalize_memory_args(args)
        self.assertFalse(normalized.use_memory)
        self.assertFalse(normalized.use_continuous_memory)
        self.assertFalse(normalized.use_text_memory)


if __name__ == "__main__":
    unittest.main()
