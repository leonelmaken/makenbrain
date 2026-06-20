"""In-memory rate limiting for MakenBrain's sensitive endpoints.

Process-local sliding-window limiter keyed by (client IP, endpoint path).
This is intentionally simple -- sufficient for a single-process local-first
deployment (`uvicorn main:app`, no multiple workers). If MakenBrain is ever
deployed with multiple workers or instances, replace the in-memory store
with a shared backend (Redis, etc.) -- out of scope for Phase 1.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

from core.audit import audit_event
from core.config import settings


class RateLimiter:
    """Sliding-window rate limiter keyed by an arbitrary string key."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> bool:
        """Register a hit for `key` and return False if the rate is exceeded.

        Args:
            key: Arbitrary bucket identifier (e.g. "{client_ip}:{path}").

        Returns:
            True if the request is allowed and has been counted, False if
            the caller already exceeded max_requests within the window.
        """
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and now - bucket[0] > self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            return True


# Limiteur partagé par tous les endpoints sensibles (agent, scheduler,
# audit, écriture mémoire). Seuils centralisés dans core/config.py.
critical_limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
)


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency rejecting requests beyond the critical rate limit.

    Raises:
        HTTPException: 429 if the client exceeded the allowed request rate
            for this endpoint within the current sliding window.
    """
    client_ip = request.client.host if request.client else "unknown"
    key = f"{client_ip}:{request.url.path}"
    if not critical_limiter.check(key):
        audit_event(
            action="rate_limit.blocked",
            tool="rate_limit",
            endpoint=request.url.path,
            result=f"ip={client_ip}",
            success=False,
        )
        raise HTTPException(
            status_code=429,
            detail="Trop de requêtes sur cet endpoint. Réessaie dans une minute.",
        )
