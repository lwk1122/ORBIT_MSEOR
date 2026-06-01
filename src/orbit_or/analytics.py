"""PostHog analytics client singleton for ORBIT.

Initializes a shared Posthog instance from environment variables.
All modules import `posthog_client` from here and call capture().
"""

import atexit
import logging
import os

logger = logging.getLogger(__name__)

_client = None


def _init_client():
    global _client
    if _client is not None:
        return _client

    token = os.getenv("POSTHOG_PROJECT_TOKEN")
    if not token:
        logger.warning(
            "[PostHog] POSTHOG_PROJECT_TOKEN not set; analytics disabled."
        )
        return None

    try:
        from posthog import Posthog

        host = os.getenv("POSTHOG_HOST")
        kwargs = {"enable_exception_autocapture": True}
        if host:
            kwargs["host"] = host
        _client = Posthog(token, **kwargs)
        atexit.register(_client.shutdown)
        logger.info("[PostHog] Analytics client initialized.")
    except Exception as exc:
        logger.warning("[PostHog] Failed to initialize analytics client: %s", exc)
        _client = None

    return _client


def get_client():
    """Return the shared PostHog client, initializing it on first call."""
    return _init_client()


def capture(distinct_id: str, event: str, properties: dict | None = None) -> None:
    """Capture an analytics event. No-ops silently if client is not configured."""
    client = get_client()
    if client is None:
        return
    try:
        client.capture(distinct_id=distinct_id, event=event, properties=properties or {})
    except Exception as exc:
        logger.debug("[PostHog] capture() failed: %s", exc)


def capture_exception(exc: Exception, distinct_id: str) -> None:
    """Capture a handled exception. No-ops silently if client is not configured."""
    client = get_client()
    if client is None:
        return
    try:
        client.capture_exception(exc, distinct_id)
    except Exception as inner:
        logger.debug("[PostHog] capture_exception() failed: %s", inner)
