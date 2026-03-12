"""Structured error classification for GUI agent action failures."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ErrorCategory(Enum):
    ELEMENT_NOT_FOUND = "element_not_found"
    PAGE_TIMEOUT = "page_timeout"
    AUTH_EXPIRED = "auth_expired"
    CAPTCHA_DETECTED = "captcha_detected"
    NAVIGATION_FAILED = "navigation_failed"
    NETWORK_ERROR = "network_error"
    DYNAMIC_CONTENT_CHANGED = "dynamic_content_changed"
    UNKNOWN = "unknown"


# Categories that are safe to retry automatically
RETRYABLE_CATEGORIES = frozenset({
    ErrorCategory.ELEMENT_NOT_FOUND,
    ErrorCategory.PAGE_TIMEOUT,
    ErrorCategory.NETWORK_ERROR,
    ErrorCategory.DYNAMIC_CONTENT_CHANGED,
    ErrorCategory.NAVIGATION_FAILED,
})

_RECOVERY_MAP = {
    ErrorCategory.ELEMENT_NOT_FOUND: "retry",
    ErrorCategory.PAGE_TIMEOUT: "refresh",
    ErrorCategory.AUTH_EXPIRED: "re_auth",
    ErrorCategory.CAPTCHA_DETECTED: "abort",
    ErrorCategory.NAVIGATION_FAILED: "retry",
    ErrorCategory.NETWORK_ERROR: "retry",
    ErrorCategory.DYNAMIC_CONTENT_CHANGED: "retry",
    ErrorCategory.UNKNOWN: "abort",
}


@dataclass
class ClassifiedError:
    category: ErrorCategory
    message: str
    is_retryable: bool
    suggested_recovery: str  # "retry" | "refresh" | "re_auth" | "back" | "abort"
    raw_exception: Optional[Exception] = field(default=None, repr=False)

    def __str__(self) -> str:
        return f"[{self.category.value}] {self.message} (recovery={self.suggested_recovery})"


def classify_error(
    exception: Exception,
    page_content: str = "",
    page_url: str = "",
) -> ClassifiedError:
    """Classify a caught exception into a structured error."""
    exc_type = type(exception).__name__
    exc_msg = str(exception).lower()

    # Playwright TimeoutError
    if "timeout" in exc_type.lower() or "timeout" in exc_msg:
        return ClassifiedError(
            category=ErrorCategory.PAGE_TIMEOUT,
            message=f"Page or element timed out: {exception}",
            is_retryable=True,
            suggested_recovery="refresh",
            raw_exception=exception,
        )

    # Element / coordinate failures
    if any(kw in exc_msg for kw in ("could not determine coordinates", "element not found", "no node found", "not visible")):
        return ClassifiedError(
            category=ErrorCategory.ELEMENT_NOT_FOUND,
            message=f"Element interaction failed: {exception}",
            is_retryable=True,
            suggested_recovery="retry",
            raw_exception=exception,
        )

    # Navigation failures (goto)
    if any(kw in exc_msg for kw in ("net::err_", "navigation", "goto")):
        return ClassifiedError(
            category=ErrorCategory.NAVIGATION_FAILED,
            message=f"Navigation failed: {exception}",
            is_retryable=True,
            suggested_recovery="retry",
            raw_exception=exception,
        )

    # Network errors
    if any(kw in exc_msg for kw in ("connection", "dns", "network", "socket", "ssl")):
        return ClassifiedError(
            category=ErrorCategory.NETWORK_ERROR,
            message=f"Network error: {exception}",
            is_retryable=True,
            suggested_recovery="retry",
            raw_exception=exception,
        )

    # Fall back to page-content heuristics if available
    if page_content:
        content_error = classify_from_page_content(page_content, page_url)
        if content_error is not None:
            content_error.raw_exception = exception
            return content_error

    # Unknown
    return ClassifiedError(
        category=ErrorCategory.UNKNOWN,
        message=f"Unclassified error ({exc_type}): {exception}",
        is_retryable=False,
        suggested_recovery="abort",
        raw_exception=exception,
    )


def classify_from_page_content(
    content: str,
    url: str = "",
) -> Optional[ClassifiedError]:
    """Inspect page content for signs of auth expiry, captcha, etc."""
    if not content:
        return None

    content_lower = content.lower()

    # CAPTCHA detection
    if any(kw in content_lower for kw in ("captcha", "recaptcha", "hcaptcha", "verify you are human")):
        return ClassifiedError(
            category=ErrorCategory.CAPTCHA_DETECTED,
            message="CAPTCHA detected on page",
            is_retryable=False,
            suggested_recovery="abort",
        )

    # Auth / login redirect
    if any(kw in content_lower for kw in ("sign in", "log in", "login", "session expired", "unauthorized")):
        if any(kw in content_lower for kw in ("<form", "<input")):
            return ClassifiedError(
                category=ErrorCategory.AUTH_EXPIRED,
                message="Authentication required — login form detected",
                is_retryable=False,
                suggested_recovery="re_auth",
            )

    # Very short content may indicate dynamic content not yet loaded
    if len(content.strip()) < 100:
        return ClassifiedError(
            category=ErrorCategory.DYNAMIC_CONTENT_CHANGED,
            message="Page content suspiciously short — possible loading issue",
            is_retryable=True,
            suggested_recovery="retry",
        )

    return None
