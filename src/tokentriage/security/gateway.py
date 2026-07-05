"""Security Gateway — first thing every request touches.

Four checks, in order:
  1. Input sanitizer     — length cap, control-char strip, payload shape
  2. Injection screen    — heuristic patterns for prompt-injection attempts;
                           flagged requests are QUARANTINED (logged), not
                           silently dropped, so operators can audit them
  3. Rate limiter        — token bucket per client key
  4. Budget precheck     — delegates to the circuit breaker (security/budget.py)

Design choice: heuristics run BEFORE any model call, so hostile input never
reaches a paid model and never costs money. Borderline input can be routed
through a future model-based second-pass screen without changing this contract.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from tokentriage import db

# Deliberately simple, auditable patterns. Not exhaustive — the point is
# defense-in-depth plus a visible quarantine trail, not a perfect filter.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(previous|prior|the|above|earlier)\s+(instructions|prompts|rules)",
    r"disregard (the|your) (system|previous) (prompt|instructions)",
    r"you are now (dan|developer mode|unfiltered)",
    r"reveal (your|the) (system prompt|instructions|api key)",
    r"print (your|the) (system prompt|initial instructions)",
    r"\bbase64\b.{0,40}\bdecode\b.{0,40}\binstructions\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class SecurityError(Exception):
    """Raised for any gateway rejection. Carries an HTTP-ish status + reason."""

    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass
class RateLimiter:
    """Token bucket per client key. Refill = limit_per_minute / 60 tokens/sec."""

    limit_per_minute: int = 60
    _buckets: dict = field(default_factory=dict)

    def check(self, client_key: str) -> None:
        now = time.time()
        tokens, last = self._buckets.get(client_key, (float(self.limit_per_minute), now))
        tokens = min(self.limit_per_minute, tokens + (now - last) * self.limit_per_minute / 60)
        if tokens < 1:
            raise SecurityError(429, "rate_limited")
        self._buckets[client_key] = (tokens - 1, now)


def sanitize(task: str, max_chars: int) -> str:
    """Strip control characters and enforce the policy length cap."""
    if not isinstance(task, str) or not task.strip():
        raise SecurityError(400, "empty_or_invalid_input")
    task = _CONTROL_CHARS.sub("", task)
    if len(task) > max_chars:
        raise SecurityError(413, f"input_exceeds_{max_chars}_chars")
    return task.strip()


def injection_screen(task: str) -> None:
    """Quarantine + reject requests matching injection heuristics."""
    for rx in _INJECTION_RE:
        if rx.search(task):
            db.quarantine(task, f"injection_pattern:{rx.pattern}")
            raise SecurityError(400, "request_quarantined_prompt_injection")
    # Extension point: borderline inputs can be classified with a strict
    # yes/no schema here while still staying pre-routing.


def gateway_check(task: str, client_key: str, policy: dict, limiter: RateLimiter) -> str:
    """Run the full gateway pipeline. Returns the sanitized task or raises."""
    sec = policy.get("security", {})
    limiter.limit_per_minute = int(sec.get("rate_limit_per_minute", 60))
    limiter.check(client_key)
    task = sanitize(task, int(sec.get("max_input_chars", 20000)))
    injection_screen(task)
    return task
