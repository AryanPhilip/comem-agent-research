"""Exponential backoff wrapper for action execution."""
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from .error_classifier import (
    ClassifiedError,
    ErrorCategory,
    RETRYABLE_CATEGORIES,
    classify_error,
)

logger = logging.getLogger("logger")


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    backoff_factor: float = 2.0
    retryable_categories: tuple = field(
        default_factory=lambda: tuple(RETRYABLE_CATEGORIES)
    )


def action_retry(
    action_fn: Callable[[], Any],
    retry_config: RetryConfig,
    action_description: str = "",
) -> Tuple[Any, Optional[ClassifiedError]]:
    """Execute *action_fn* with exponential back-off on retryable errors.

    Returns ``(result, None)`` on success or ``(None, last_classified_error)``
    after all retries are exhausted.
    """
    last_error: Optional[ClassifiedError] = None
    delay = retry_config.base_delay

    for attempt in range(1 + retry_config.max_retries):
        try:
            result = action_fn()
            return result, None
        except Exception as exc:
            classified = classify_error(exc)
            last_error = classified

            retryable = classified.category in retry_config.retryable_categories
            remaining = retry_config.max_retries - attempt

            label = action_description or "action"
            logger.info(
                f"[Retry] {label} attempt {attempt + 1} failed: {classified} "
                f"(retryable={retryable}, remaining={remaining})"
            )

            if not retryable or remaining <= 0:
                break

            time.sleep(delay)
            delay = min(delay * retry_config.backoff_factor, retry_config.max_delay)

    return None, last_error
