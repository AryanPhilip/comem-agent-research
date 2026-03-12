"""Lightweight planner, verifier, and structured page-state helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
from urllib.parse import urlparse


def _infer_page_type(url: str) -> str:
    lowered = url.lower()
    if "search" in lowered:
        return "search"
    if any(token in lowered for token in ("product", "item", "detail")):
        return "detail"
    if any(token in lowered for token in ("cart", "checkout", "booking")):
        return "transaction"
    if any(token in lowered for token in ("wiki", "article", "docs", "arxiv")):
        return "content"
    return "browse"


@dataclass
class StructuredPageState:
    current_url: str = ""
    page_type: str = "browse"
    salient_controls: list[str] = field(default_factory=list)
    current_subgoal: str = ""
    retry_count: int = 0
    recent_action_summary: str = ""
    failure_state: str = ""

    def to_prompt(self) -> str:
        controls = ", ".join(self.salient_controls) if self.salient_controls else "unknown"
        return (
            "Structured page state:\n"
            f"- URL: {self.current_url or 'unknown'}\n"
            f"- Page type: {self.page_type}\n"
            f"- Salient controls: {controls}\n"
            f"- Current subgoal: {self.current_subgoal or 'unspecified'}\n"
            f"- Retry count: {self.retry_count}\n"
            f"- Recent actions: {self.recent_action_summary or 'none'}\n"
            f"- Failure state: {self.failure_state or 'none'}"
        )


@dataclass
class VerificationResult:
    needs_refresh: bool = False
    failure_state: str = ""
    reflection: str = ""
    verifier_note: str = ""


class TaskPlanner:
    """Cheap planner that converts agent state into a short subgoal."""

    def plan(self, intent: str, state: StructuredPageState, action_history: Sequence[str]) -> str:
        if not action_history:
            return "Find the primary entry point for the task and gather the first decisive state change."
        last_action = action_history[-1].lower()
        if "type" in last_action or "search" in last_action:
            return "Validate the search submission and inspect the most relevant result."
        if "scroll" in last_action:
            return "Look for a better candidate element before taking another interaction."
        if state.failure_state:
            return "Recover from the failed path by choosing a different control or page branch."
        if state.page_type == "detail":
            return "Extract the needed evidence from the current detail view or finish the task."
        return "Progress toward the task outcome while avoiding repeated actions."


class StepVerifier:
    """Heuristic verifier for refresh and reflection triggers."""

    WRONG_PAGE_TOKENS = {"google", "bing", "duckduckgo", "search"}

    def verify(
        self,
        intent: str,
        state: StructuredPageState,
        action_history: Sequence[str],
        response_history: Sequence[str],
        site: str = "",
    ) -> VerificationResult:
        if len(action_history) >= 3 and len(set(action_history[-3:])) == 1:
            return VerificationResult(
                needs_refresh=True,
                failure_state="repeated_action_loop",
                reflection="The last action repeated without progress. Choose a different control or navigate to a new page state.",
                verifier_note="Verifier flagged repeated action loop.",
            )

        if response_history:
            last_response = response_history[-1].lower()
            if any(token in last_response for token in ("not found", "cannot", "no results", "missing")):
                return VerificationResult(
                    needs_refresh=True,
                    failure_state="missing_element",
                    reflection="The target element was not found. Adjust the target element or recover by changing the page context.",
                    verifier_note="Verifier flagged missing element failure.",
                )

        expected_site_tokens = {token for token in site.lower().replace("-", " ").split() if token}
        current_tokens = {token for token in urlparse(state.current_url).netloc.lower().replace("www.", "").replace(".", " ").split() if token}
        if (
            expected_site_tokens
            and current_tokens
            and not expected_site_tokens.intersection(current_tokens)
            and current_tokens.intersection(self.WRONG_PAGE_TOKENS)
        ):
            return VerificationResult(
                needs_refresh=True,
                failure_state="wrong_page",
                reflection="You are likely on the wrong page. Recover by navigating back to the target site or opening a more relevant result.",
                verifier_note="Verifier flagged page drift.",
            )

        return VerificationResult()


class ReflectionBuffer:
    """Small priority buffer for corrective reflections."""

    def __init__(self) -> None:
        self._entries: list[str] = []

    def reset(self) -> None:
        self._entries.clear()

    def push(self, reflection: str) -> None:
        if not reflection:
            return
        if reflection in self._entries:
            self._entries.remove(reflection)
        self._entries.insert(0, reflection)
        del self._entries[3:]

    def top(self) -> str:
        return self._entries[0] if self._entries else ""


def build_structured_page_state(
    current_url: str,
    action_history: Sequence[str],
    current_subgoal: str,
    failure_state: str,
) -> StructuredPageState:
    recent_actions = " | ".join(action_history[-3:])
    retry_count = 0
    if action_history:
        last_action = action_history[-1]
        retry_count = sum(1 for action in reversed(action_history) if action == last_action)

    return StructuredPageState(
        current_url=current_url,
        page_type=_infer_page_type(current_url),
        salient_controls=[],
        current_subgoal=current_subgoal,
        retry_count=retry_count,
        recent_action_summary=recent_actions,
        failure_state=failure_state,
    )

