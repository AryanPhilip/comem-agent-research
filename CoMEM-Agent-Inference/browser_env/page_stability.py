"""Wait for DOM/network stability instead of fixed sleeps."""
import logging
import time

logger = logging.getLogger("logger")


def wait_for_stable(
    page,
    timeout_ms: int = 5000,
    idle_threshold_ms: int = 500,
    fallback_sleep: float = 2.5,
) -> bool:
    """Wait until the page has no network activity and no DOM mutations.

    Uses a CDP session to track in-flight requests and injects a
    MutationObserver via ``Runtime.evaluate`` to detect DOM changes.

    Returns ``True`` if stability was detected within *timeout_ms*,
    ``False`` if the timeout was reached (in which case we fall back
    to the original sleep behaviour).
    """
    try:
        cdp = page.context.new_cdp_session(page)
    except Exception:
        # CDP not available (e.g. headless mode quirks) — fall back
        time.sleep(fallback_sleep)
        return False

    try:
        # Inject a small JS snippet that tracks DOM mutations
        page.evaluate("""() => {
            window.__stabilityLastMutation = Date.now();
            if (!window.__stabilityObserver) {
                window.__stabilityObserver = new MutationObserver(() => {
                    window.__stabilityLastMutation = Date.now();
                });
                window.__stabilityObserver.observe(document.body || document.documentElement, {
                    childList: true, subtree: true, attributes: true
                });
            }
        }""")

        # Track network via CDP
        inflight = {"count": 0}

        def _on_request_will_be_sent(params):
            inflight["count"] += 1

        def _on_loading_finished(params):
            inflight["count"] = max(inflight["count"] - 1, 0)

        def _on_loading_failed(params):
            inflight["count"] = max(inflight["count"] - 1, 0)

        cdp.on("Network.requestWillBeSent", _on_request_will_be_sent)
        cdp.on("Network.loadingFinished", _on_loading_finished)
        cdp.on("Network.loadingFailed", _on_loading_failed)
        cdp.send("Network.enable")

        poll_interval = 0.1  # seconds
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        idle_since: float | None = None

        while time.monotonic() < deadline:
            network_idle = inflight["count"] == 0

            # Check DOM mutation recency
            try:
                last_mut = page.evaluate("() => window.__stabilityLastMutation || 0")
            except Exception:
                last_mut = 0
            now_ms = time.time() * 1000
            dom_idle = (now_ms - last_mut) > idle_threshold_ms if last_mut else False

            if network_idle and dom_idle:
                if idle_since is None:
                    idle_since = time.monotonic()
                elif (time.monotonic() - idle_since) * 1000 >= idle_threshold_ms:
                    logger.debug("[PageStability] Page stable — proceeding")
                    return True
            else:
                idle_since = None

            time.sleep(poll_interval)

        # Timeout reached — fall back
        logger.debug("[PageStability] Timeout reached — using fallback sleep")
        time.sleep(fallback_sleep)
        return False

    except Exception as exc:
        logger.debug(f"[PageStability] Error during stability wait: {exc} — using fallback sleep")
        time.sleep(fallback_sleep)
        return False
    finally:
        try:
            cdp.detach()
        except Exception:
            pass
