import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.getcwd(), "CoMEM-Agent-Inference"))

from memory.runtime import (  # noqa: E402
    MemoryBundle,
    MemoryQuery,
    NullMemoryInjector,
    QwenContinuousMemoryInjector,
    TrajectoryRecord,
    score_memory_candidate,
)


class MemoryRuntimeTest(unittest.TestCase):
    def test_reranker_rewards_success_site_and_shorter_paths(self):
        query = MemoryQuery(
            task="book a hotel in paris",
            dataset="webvoyager",
            domain="Booking",
            site="booking.com",
            current_url="https://www.booking.com/searchresults.html",
            failure_state="wrong_page",
        )
        strong_record = TrajectoryRecord(
            file_path="good.jsonl",
            task_description="book a hotel in paris",
            prefixed_query="webvoyager_Booking: book a hotel in paris",
            dataset="webvoyager",
            domain="Booking",
            site="booking.com",
            success=True,
            step_count=4,
            timestamp="2026-03-09T10:00:00+00:00",
            failure_tags=["wrong_page"],
        )
        weak_record = TrajectoryRecord(
            file_path="bad.jsonl",
            task_description="book a hotel in paris",
            prefixed_query="webvoyager_Booking: book a hotel in paris",
            dataset="webvoyager",
            domain="Booking",
            site="google.com",
            success=False,
            step_count=12,
            timestamp="2025-01-01T10:00:00+00:00",
            failure_tags=["missing_element"],
        )

        strong_score = score_memory_candidate(query, strong_record, similarity_score=0.7)
        weak_score = score_memory_candidate(query, weak_record, similarity_score=0.7)
        self.assertGreater(strong_score, weak_score)

    def test_injectors_match_expected_modes(self):
        bundle = MemoryBundle(
            prompt_text="EXAMPLE: test",
            experience_texts=[[{"name": "click", "arguments": {"reasoning": "go", "description": ""}}]],
            experience_images=[[""]],
            file_id_list=["demo"],
        )
        null_kwargs = NullMemoryInjector().prepare_chat_kwargs(bundle, memory_token_budget=8)
        qwen_kwargs = QwenContinuousMemoryInjector().prepare_chat_kwargs(bundle, memory_token_budget=8)

        self.assertEqual(null_kwargs, {})
        self.assertEqual(qwen_kwargs["memory_token_budget"], 8)
        self.assertEqual(qwen_kwargs["file_id_list"], ["demo"])


if __name__ == "__main__":
    unittest.main()
