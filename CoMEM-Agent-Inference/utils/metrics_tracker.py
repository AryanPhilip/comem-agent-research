"""Utilities for tracking inference metrics across evaluation runs."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TaskMetrics:
    """Container for metrics collected for a single evaluation task."""

    task_id: str
    success: bool
    start_time: float
    end_time: float
    steps: Optional[int] = None
    tokens: Optional[int] = None
    final_url: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def duration(self) -> float:
        return self.end_time - self.start_time


class InferenceMetricsTracker:
    """Tracks per-task and aggregate metrics for inference runs."""

    def __init__(self, result_dir: Optional[str], run_metadata: Optional[Dict[str, Any]] = None):
        self.result_dir = Path(result_dir) if result_dir else None
        self.run_metadata = run_metadata or {}
        self._active_tasks: Dict[str, Dict[str, Any]] = {}
        self._completed_tasks: List[TaskMetrics] = []

        if self.result_dir:
            self.result_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------
    def start_task(self, task_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register the start of a task."""

        self._active_tasks[task_id] = {
            "start_time": time.time(),
            "metadata": metadata or {},
        }

    def end_task(
        self,
        task_id: str,
        *,
        success: bool,
        steps: Optional[int] = None,
        tokens: Optional[int] = None,
        final_url: Optional[str] = None,
        score: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finalize a task and store the collected metrics."""

        task = self._active_tasks.pop(task_id, None)
        if task is None:
            # Task was not started (or already finalized); skip gracefully.
            return

        end_time = time.time()
        metadata = {**task.get("metadata", {}), **(extra_metadata or {})}

        self._completed_tasks.append(
            TaskMetrics(
                task_id=task_id,
                success=success,
                start_time=task["start_time"],
                end_time=end_time,
                steps=steps,
                tokens=tokens,
                final_url=final_url,
                score=score,
                metadata=metadata,
            )
        )

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _aggregate_numeric_field(self, field_name: str) -> Dict[str, Optional[float]]:
        values = [getattr(task, field_name) for task in self._completed_tasks if getattr(task, field_name) is not None]
        if not values:
            return {"total": None, "average": None, "min": None, "max": None}

        total = float(sum(values))
        return {
            "total": total,
            "average": float(statistics.mean(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    def _aggregate_duration(self) -> Dict[str, Optional[float]]:
        durations = [task.duration() for task in self._completed_tasks]
        if not durations:
            return {"total": None, "average": None, "min": None, "max": None}

        total = float(sum(durations))
        return {
            "total": total,
            "average": float(statistics.mean(durations)),
            "min": float(min(durations)),
            "max": float(max(durations)),
        }

    def _average_metadata_field(self, field_name: str) -> Optional[float]:
        values = [
            float(task.metadata[field_name])
            for task in self._completed_tasks
            if task.metadata.get(field_name) is not None
        ]
        if not values:
            return None
        return float(statistics.mean(values))

    def _rate_from_metadata(self, numerator_field: str, denominator: str, as_presence: bool = False) -> Optional[float]:
        if not self._completed_tasks:
            return None

        if denominator == "steps":
            total_denominator = sum(task.steps or 0 for task in self._completed_tasks)
        elif denominator == "tasks_with_memory":
            total_denominator = sum(
                1
                for task in self._completed_tasks
                if self.run_metadata.get("use_memory") or task.metadata.get("memory_refreshes")
            )
        else:
            total_denominator = len(self._completed_tasks)

        if not total_denominator:
            return None

        if as_presence:
            total_numerator = sum(
                1.0 for task in self._completed_tasks if float(task.metadata.get(numerator_field, 0) or 0) > 0
            )
        else:
            total_numerator = sum(float(task.metadata.get(numerator_field, 0)) for task in self._completed_tasks)
        return total_numerator / float(total_denominator)

    def _success_by_horizon_bucket(self) -> Dict[str, Dict[str, Optional[float]]]:
        buckets = {
            "short": [],
            "medium": [],
            "long": [],
        }
        for task in self._completed_tasks:
            steps = task.steps or 0
            if steps <= 5:
                buckets["short"].append(task)
            elif steps <= 10:
                buckets["medium"].append(task)
            else:
                buckets["long"].append(task)

        summary = {}
        for bucket_name, tasks in buckets.items():
            total = len(tasks)
            successes = sum(1 for task in tasks if task.success)
            summary[bucket_name] = {
                "tasks": total,
                "success_rate": (successes / total) if total else None,
            }
        return summary

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_summary(self) -> Dict[str, Any]:
        total_tasks = len(self._completed_tasks)
        successes = sum(1 for task in self._completed_tasks if task.success)
        success_rate = (successes / total_tasks) if total_tasks else None

        summary = {
            "run_metadata": self.run_metadata,
            "totals": {
                "tasks": total_tasks,
                "successes": successes,
                "failures": total_tasks - successes if total_tasks else 0,
                "success_rate": success_rate,
            },
            "durations": self._aggregate_duration(),
            "steps": self._aggregate_numeric_field("steps"),
            "tokens": self._aggregate_numeric_field("tokens"),
            "derived_metrics": {
                "average_repeated_actions": self._average_metadata_field("repeated_action_count"),
                "repeated_action_rate": self._rate_from_metadata("repeated_action_count", "steps"),
                "verifier_intervention_rate": self._rate_from_metadata("verifier_interventions", "tasks", as_presence=True),
                "retrieval_hit_rate": self._rate_from_metadata("memory_hits", "tasks_with_memory", as_presence=True),
            },
            "performance_by_horizon_bucket": self._success_by_horizon_bucket(),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "success": task.success,
                    "duration": task.duration(),
                    "steps": task.steps,
                    "tokens": task.tokens,
                    "final_url": task.final_url,
                    "score": task.score,
                    "metadata": task.metadata,
                }
                for task in self._completed_tasks
            ],
        }

        return summary

    def save_summary(self) -> Dict[str, Any]:
        summary = self.get_summary()

        if self.result_dir is None:
            return summary

        json_path = self.result_dir / "metrics_summary.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        # Optional markdown summary for quick inspection
        md_path = self.result_dir / "metrics_summary.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Inference Metrics Summary\n\n")
            if summary["totals"]["tasks"]:
                success_rate_pct = (
                    summary["totals"]["success_rate"] * 100.0
                    if summary["totals"]["success_rate"] is not None
                    else None
                )
                f.write(
                    f"- Tasks processed: {summary['totals']['tasks']}\n"
                    f"- Successes: {summary['totals']['successes']}\n"
                    f"- Failures: {summary['totals']['failures']}\n"
                )
                if success_rate_pct is not None:
                    f.write(f"- Success rate: {success_rate_pct:.2f}%\n")

            duration_avg = summary["durations"]["average"]
            if duration_avg is not None:
                f.write(f"- Average duration: {duration_avg:.2f}s\n")

            steps_avg = summary["steps"]["average"]
            if steps_avg is not None:
                f.write(f"- Average steps: {steps_avg:.2f}\n")

            tokens_avg = summary["tokens"]["average"]
            if tokens_avg is not None:
                f.write(f"- Average tokens: {tokens_avg:.2f}\n")

            repeated_action_rate = summary["derived_metrics"]["repeated_action_rate"]
            if repeated_action_rate is not None:
                f.write(f"- Repeated action rate: {repeated_action_rate:.4f}\n")

            verifier_rate = summary["derived_metrics"]["verifier_intervention_rate"]
            if verifier_rate is not None:
                f.write(f"- Verifier intervention rate: {verifier_rate:.4f}\n")

            retrieval_hit_rate = summary["derived_metrics"]["retrieval_hit_rate"]
            if retrieval_hit_rate is not None:
                f.write(f"- Retrieval hit rate: {retrieval_hit_rate:.4f}\n")

            f.write("\n## Horizon Buckets\n\n")
            for bucket_name, bucket_summary in summary["performance_by_horizon_bucket"].items():
                success_rate = bucket_summary["success_rate"]
                if success_rate is None:
                    f.write(f"- {bucket_name}: 0 tasks\n")
                else:
                    f.write(
                        f"- {bucket_name}: {bucket_summary['tasks']} tasks, success rate {success_rate * 100.0:.2f}%\n"
                    )

            if summary["tasks"]:
                f.write("\n## Tasks\n\n")
                f.write("| Task ID | Success | Duration (s) | Steps | Tokens |\n")
                f.write("|---------|---------|--------------|-------|--------|\n")
                for task in summary["tasks"]:
                    duration = f"{task['duration']:.2f}" if task["duration"] is not None else "-"
                    steps = task["steps"] if task["steps"] is not None else "-"
                    tokens = task["tokens"] if task["tokens"] is not None else "-"
                    f.write(
                        f"| {task['task_id']} | {task['success']} | {duration} | {steps} | {tokens} |\n"
                    )

        return summary
