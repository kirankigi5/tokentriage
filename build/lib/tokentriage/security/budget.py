"""Budget circuit breaker.

Reads the spend ledger (db.budget) and enforces the policy's daily cap.
Behavior at the cap is *graceful degradation*, not a hard outage:

  - Expensive tiers (T2+) are disallowed once the cap is hit.
  - T1 remains available (nearly free) so the service keeps answering.
  - If even T1 is disallowed by policy for the task type, the caller
    receives a clean 429 with a machine-readable reason.

This is deliberately a *pre-dispatch* check plus a *post-dispatch* ledger
write — a single request can slightly overshoot the cap, which we accept
and document rather than adding locking complexity for a demo-scale system.
"""
from __future__ import annotations

from tokentriage import db
from tokentriage.security.gateway import SecurityError

# Tiers considered "expensive" for breaker purposes.
_EXPENSIVE = {"T2", "T3", "T4", "T5", "T6", "T7"}


def allowed_tiers(policy: dict, candidate_tiers: list[str]) -> list[str]:
    """Filter candidate tiers by remaining daily budget."""
    cap = float(policy["default"]["daily_budget_usd"])
    spent = db.spend_today_usd()
    if spent < cap:
        return candidate_tiers
    remaining = [t for t in candidate_tiers if t not in _EXPENSIVE]
    if not remaining:
        raise SecurityError(429, f"daily_budget_exhausted_({spent:.4f}usd/{cap:.2f}usd)")
    return remaining


def record(tier: str, cost_usd: float) -> None:
    """Post-dispatch ledger write."""
    if cost_usd > 0:
        db.record_spend(tier, cost_usd)
