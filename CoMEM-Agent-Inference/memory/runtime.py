"""Shared runtime primitives for retrieval, condensation, and injection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence
from urllib.parse import urlparse

try:
    import faiss
except ImportError:  # pragma: no cover - optional for pure-logic tests
    faiss = None
import numpy as np
import json
import re


@dataclass
class TrajectoryRecord:
    """Normalized memory record used by the runtime retriever."""

    file_path: str
    task_description: str
    prefixed_query: str
    dataset: str = ""
    domain: str = ""
    site: str = ""
    model: str = ""
    success: bool = True
    step_count: int = 0
    timestamp: str = ""
    failure_tags: list[str] = field(default_factory=list)
    base64_image: Optional[str] = None
    actions: list[dict] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    semantic_summary: str = ""
    reflection_summary: str = ""
    example_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TrajectoryRecord":
        return cls(**payload)


@dataclass
class MemoryQuery:
    """Query state for dynamic retrieval."""

    task: str
    dataset: str = ""
    domain: str = ""
    site: str = ""
    current_url: str = ""
    action_history: list[str] = field(default_factory=list)
    failure_state: str = ""
    current_image: Optional[str] = None


@dataclass
class MemorySearchResult:
    record: TrajectoryRecord
    similarity: float
    rerank_score: float


@dataclass
class MemoryBundle:
    """Prompt-facing memory payload."""

    prompt_text: str = ""
    experience_texts: list[list[dict]] = field(default_factory=list)
    experience_images: list[list[str]] = field(default_factory=list)
    file_id_list: list[str] = field(default_factory=list)
    semantic_notes: list[str] = field(default_factory=list)
    selected_records: list[TrajectoryRecord] = field(default_factory=list)

    def has_continuous_memory(self) -> bool:
        return bool(self.experience_texts and any(self.experience_texts))


class TrajectoryCondenser:
    """Condense stored trajectories into episodic and semantic memories."""

    def normalize_action(self, response: object) -> Optional[dict]:
        if isinstance(response, list) and response:
            response = response[0]
        if isinstance(response, dict) and "content" in response:
            response = response["content"]
        if not isinstance(response, str):
            return None

        parsed = _parse_action_json(response)
        if not isinstance(parsed, dict):
            return None
        action_json = parsed.get("function_call", {})
        if not isinstance(action_json, dict):
            return None

        name = (
            action_json.get("name")
            or action_json.get("action")
            or action_json.get("action_type")
            or action_json.get("type")
        )
        if not name and action_json:
            first_value = next(iter(action_json.values()))
            if isinstance(first_value, dict):
                name = next(iter(action_json.keys()))
                action_json = {"name": name, "arguments": first_value}

        if not name:
            return None

        arguments = action_json.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {"reasoning": str(arguments)}

        reasoning = arguments.get("reasoning", "")
        description = (
            arguments.get("description")
            or arguments.get("field_description")
            or arguments.get("answer")
            or arguments.get("query")
            or ""
        )

        return {
            "name": str(name),
            "arguments": {
                "reasoning": str(reasoning),
                "description": str(description),
            },
        }

    def extract_actions_and_images(self, rounds: Sequence[dict]) -> tuple[list[dict], list[str]]:
        actions: list[dict] = []
        images: list[str] = []
        previous_name = None

        for round_item in rounds:
            image = self.extract_base64_image(round_item)
            action = self.normalize_action(round_item.get("response"))
            if not action:
                continue
            if action["name"] == previous_name:
                continue
            actions.append(action)
            images.append(image or "")
            previous_name = action["name"]

        return actions, images

    def extract_base64_image(self, round_item: dict) -> Optional[str]:
        messages = round_item.get("messages", [])
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                image_url = item.get("image_url", {})
                if image_url.get("url"):
                    return image_url["url"]
        return None

    def summarize_actions(self, actions: Sequence[dict]) -> str:
        if not actions:
            return "No parsed actions available."
        action_names = [action["name"] for action in actions]
        counts = Counter(action_names)
        dominant = ", ".join(f"{name} x{count}" for name, count in counts.most_common(3))
        ordered = " -> ".join(action_names[:6])
        return f"Workflow: {ordered}. Dominant actions: {dominant}."

    def summarize_reflection(
        self,
        task_description: str,
        success: bool,
        failure_tags: Sequence[str],
        actions: Sequence[dict],
    ) -> str:
        if success:
            return f"Successful pattern for '{task_description}': favor the shortest path and stop after verification."
        failure_hint = ", ".join(failure_tags) if failure_tags else "state drift"
        if actions:
            last_action = actions[min(len(actions) - 1, 2)]["name"]
            return (
                f"Failed attempt for '{task_description}': avoid {last_action} when the page stops progressing; "
                f"recover from {failure_hint} by changing page context or target element."
            )
        return f"Failed attempt for '{task_description}': recover from {failure_hint} by changing strategy."

    def build_example_text(
        self,
        task_description: str,
        semantic_summary: str,
        reflection_summary: str,
        actions: Sequence[dict],
    ) -> str:
        lines = [f"EXAMPLE: {task_description}", semantic_summary, f"REFLECTION: {reflection_summary}"]
        if actions:
            lines.append("KEY ACTIONS:")
        for action in actions[:6]:
            reasoning = action.get("arguments", {}).get("reasoning", "")
            lines.append(f"- {action['name']}: {reasoning}")
        return "\n".join(lines)


def build_query_text(query: MemoryQuery) -> str:
    """Structured retrieval query string."""
    parts = [query.task.strip()]
    if query.dataset or query.domain:
        parts.append(f"dataset={query.dataset} domain={query.domain}".strip())
    if query.site:
        parts.append(f"site={query.site}")
    if query.current_url:
        parts.append(f"url={query.current_url}")
    if query.action_history:
        parts.append("recent_actions=" + " | ".join(query.action_history[-3:]))
    if query.failure_state:
        parts.append(f"failure_state={query.failure_state}")
    return " || ".join(part for part in parts if part)


def _host_tokens(value: str) -> set[str]:
    if not value:
        return set()
    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = parsed.netloc or parsed.path
    tokens = {token for token in host.replace("www.", "").replace("-", ".").split(".") if token}
    return tokens


def _recency_bonus(timestamp: str) -> float:
    if not timestamp:
        return 0.0
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = max((datetime.now(timezone.utc) - parsed).days, 0)
    if age_days <= 7:
        return 0.05
    if age_days <= 30:
        return 0.03
    if age_days <= 180:
        return 0.01
    return 0.0


def score_memory_candidate(
    query: MemoryQuery,
    record: TrajectoryRecord,
    similarity_score: float,
    duplicate_penalty: float = 0.0,
) -> float:
    """Deterministic reranker used by tests and runtime."""
    score = float(similarity_score)
    score += 0.15 if record.success else -0.05

    if query.domain and record.domain == query.domain:
        score += 0.08
    if query.dataset and record.dataset == query.dataset:
        score += 0.04

    query_hosts = _host_tokens(query.current_url) | _host_tokens(query.site)
    record_hosts = _host_tokens(record.site)
    if query_hosts and record_hosts and query_hosts.intersection(record_hosts):
        score += 0.08

    if record.step_count:
        score += max(0.0, 0.08 - max(record.step_count - 3, 0) * 0.01)

    if query.failure_state:
        lowered_failure = query.failure_state.lower()
        if any(lowered_failure in tag.lower() for tag in record.failure_tags):
            score += 0.06

    score += _recency_bonus(record.timestamp)
    score -= duplicate_penalty
    return score


class MemoryRetriever:
    """Hybrid episodic and semantic retriever backed by FAISS."""

    def __init__(
        self,
        records: Sequence[TrajectoryRecord],
        faiss_index,
        embeddings: Optional[np.ndarray],
        clip_similarity,
        multimodal: bool,
    ) -> None:
        self.records = list(records)
        self.faiss_index = faiss_index
        self.embeddings = embeddings
        self.clip_similarity = clip_similarity
        self.multimodal = multimodal

    def _get_query_embedding(self, query: MemoryQuery) -> np.ndarray:
        query_text = build_query_text(query)
        if self.multimodal:
            if query.current_image:
                embedding = self.clip_similarity.get_multimodal_embeddings([query_text], [query.current_image])
            else:
                text_embedding = self.clip_similarity.get_text_embeddings([query_text])
                zero_image_embedding = np.zeros_like(text_embedding)
                embedding = np.concatenate([text_embedding, zero_image_embedding], axis=1)
        else:
            embedding = self.clip_similarity.get_text_embeddings([query_text])

        if faiss is not None:
            faiss.normalize_L2(embedding)
        else:
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            embedding = embedding / np.clip(norms, 1e-12, None)
        return embedding.astype("float32")

    def retrieve(
        self,
        query: MemoryQuery,
        top_k: int,
        dataset_filter: Optional[str] = None,
        domain_filter: Optional[str] = None,
    ) -> list[MemorySearchResult]:
        if not self.records or self.faiss_index is None:
            return []

        candidate_k = min(len(self.records), max(top_k * 4, top_k + 5))
        similarities, indices = self.faiss_index.search(self._get_query_embedding(query), candidate_k)

        results: list[MemorySearchResult] = []
        seen_task_keys: Counter[str] = Counter()

        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            record = self.records[idx]
            if dataset_filter and record.dataset != dataset_filter:
                continue
            if domain_filter and record.domain != domain_filter:
                continue
            if query.task and query.task.strip() == record.task_description.strip():
                continue

            duplicate_penalty = seen_task_keys[record.task_description] * 0.12
            rerank_score = score_memory_candidate(query, record, similarity, duplicate_penalty=duplicate_penalty)
            results.append(
                MemorySearchResult(
                    record=record,
                    similarity=float(similarity),
                    rerank_score=rerank_score,
                )
            )
            seen_task_keys[record.task_description] += 1

        results.sort(key=lambda item: item.rerank_score, reverse=True)
        return results[:top_k]


class MemoryInjector(ABC):
    """Model-specific continuous-memory injector."""

    @abstractmethod
    def prepare_chat_kwargs(self, bundle: MemoryBundle, memory_token_budget: int) -> dict:
        raise NotImplementedError


class NullMemoryInjector(MemoryInjector):
    def prepare_chat_kwargs(self, bundle: MemoryBundle, memory_token_budget: int) -> dict:
        return {}


class QwenContinuousMemoryInjector(MemoryInjector):
    def prepare_chat_kwargs(self, bundle: MemoryBundle, memory_token_budget: int) -> dict:
        if not bundle.has_continuous_memory():
            return {}
        return {
            "experience_texts": bundle.experience_texts,
            "experience_images": bundle.experience_images,
            "file_id_list": bundle.file_id_list,
            "memory_token_budget": memory_token_budget,
        }


def build_memory_runtime_payload(
    memory_mode: str,
    bundle: MemoryBundle,
    fallback_prompt_text: str,
    memory_token_budget: int,
    reflection_note: str = "",
) -> dict:
    """Build prompt text and continuous-memory kwargs for a given mode."""
    if memory_mode == "text":
        prompt_text = bundle.prompt_text or fallback_prompt_text
        chat_kwargs = {}
    elif memory_mode == "continuous":
        prompt_text = "\n".join(bundle.semantic_notes[:2]) if bundle.semantic_notes else fallback_prompt_text
        chat_kwargs = QwenContinuousMemoryInjector().prepare_chat_kwargs(bundle, memory_token_budget)
    elif memory_mode == "hybrid":
        prompt_text = bundle.prompt_text or fallback_prompt_text
        chat_kwargs = QwenContinuousMemoryInjector().prepare_chat_kwargs(bundle, memory_token_budget)
    else:
        prompt_text = fallback_prompt_text
        chat_kwargs = {}

    if reflection_note:
        prompt_text = (prompt_text + "\n\nCORRECTIVE REFLECTION:\n" + reflection_note).strip()

    return {
        "prompt_text": prompt_text,
        "chat_kwargs": chat_kwargs,
    }


def _parse_action_json(message: str):
    pattern = r"Action:\s*(\{.*\})"
    match = re.search(pattern, message)
    if match:
        try:
            return {"function_call": json.loads(match.group(1))}
        except Exception:
            return message

    pattern = r"```json\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, message)
    if matches:
        try:
            return {"function_call": json.loads(matches[0])}
        except Exception:
            pass

    if "```json" in message:
        message = message.split("```json")[1].split("```")[0].strip().strip("\n").strip("\\n")
        try:
            return {"function_call": json.loads(message)}
        except Exception:
            return message

    try:
        action_json = json.loads(message)
        if isinstance(action_json, dict) and "name" in action_json and "arguments" in action_json:
            return {"function_call": action_json}
    except Exception:
        return message
    return message
