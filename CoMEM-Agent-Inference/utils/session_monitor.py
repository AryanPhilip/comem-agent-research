"""Session health tracking across agent steps with recovery suggestions."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .error_classifier import ClassifiedError, ErrorCategory


class SessionHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"       # recoverable failures accumulating
    CRITICAL = "critical"       # needs intervention
    FAILED = "failed"           # abort task


_DEFAULT_CONFIG = {
    "degraded_threshold": 2,
    "critical_threshold": 4,
    "failed_threshold": 6,
}


@dataclass
class SessionState:
    health: SessionHealth = SessionHealth.HEALTHY
    consecutive_errors: int = 0
    total_errors: int = 0
    last_error: Optional[ClassifiedError] = None
    last_url: str = ""
    step_count: int = 0


class SessionMonitor:
    """Tracks session health across steps and generates recovery suggestions."""

    def __init__(self, config: Optional[dict] = None):
        cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self._degraded_threshold: int = cfg["degraded_threshold"]
        self._critical_threshold: int = cfg["critical_threshold"]
        self._failed_threshold: int = cfg["failed_threshold"]
        self.state = SessionState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_step(
        self, error: Optional[ClassifiedError], current_url: str
    ) -> SessionHealth:
        """Record the result of an action step and return updated health."""
        self.state.step_count += 1
        self.state.last_url = current_url

        if error is None:
            # Successful step — decay consecutive error count
            self.state.consecutive_errors = max(self.state.consecutive_errors - 1, 0)
        else:
            self.state.consecutive_errors += 1
            self.state.total_errors += 1
            self.state.last_error = error

        # Determine health level
        ce = self.state.consecutive_errors
        if ce >= self._failed_threshold:
            self.state.health = SessionHealth.FAILED
        elif ce >= self._critical_threshold:
            self.state.health = SessionHealth.CRITICAL
        elif ce >= self._degraded_threshold:
            self.state.health = SessionHealth.DEGRADED
        else:
            self.state.health = SessionHealth.HEALTHY

        return self.state.health

    def get_recovery_suggestion(self) -> str:
        """Return a high-level recovery action based on current state."""
        if self.state.health == SessionHealth.HEALTHY:
            return "continue"
        if self.state.last_error is None:
            return "continue"

        return self.state.last_error.suggested_recovery

    def get_error_context_for_agent(self) -> str:
        """Return a human-readable summary suitable for injection into the LLM prompt."""
        if self.state.health == SessionHealth.HEALTHY:
            return ""

        parts = [
            f"Session health: {self.state.health.value}.",
            f"Consecutive errors: {self.state.consecutive_errors} "
            f"(total: {self.state.total_errors}).",
        ]
        if self.state.last_error:
            parts.append(
                f"Last error: {self.state.last_error.category.value} — "
                f"{self.state.last_error.message}"
            )
        suggestion = self.get_recovery_suggestion()
        if suggestion != "continue":
            parts.append(f"Suggested recovery: {suggestion}.")

        return " ".join(parts)

    def reset(self) -> None:
        """Reset state for a new task."""
        self.state = SessionState()
