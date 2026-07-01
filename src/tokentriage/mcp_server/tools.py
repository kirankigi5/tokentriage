"""MCP tool implementations.

These functions are the single source of truth for pricing, benchmarks,
budget, and decision stats. They are exposed two ways:

  1. In-process, by the orchestrator (direct import) — zero-latency path.
  2. Over MCP (server.py) — so ANY MCP-compatible agent, not just ours,
     can consume TokenTriage's cost intelligence. That interoperability
     is the reason MCP exists, and the architectural point of this module.
"""
from __future__ import annotations

from tokentriage import db
from tokentriage.models.registry import TIERS


def get_model_pricing(tier: str) -> dict:
    """USD per 1M tokens for a tier. Agents ask; they never hardcode prices."""
    t = TIERS[tier]
    return {"tier": t.tier, "model_id": t.model_id,
            "input_usd_per_m": t.input_usd_per_m,
            "output_usd_per_m": t.output_usd_per_m}


def get_accuracy_benchmark(tier: str, task_type: str) -> float:
    """Measured accuracy for (tier, task_type). Backed by the learning store —
    seeded values initially, overwritten by `tokentriage tune` from real
    verifier feedback."""
    return db.get_benchmark(tier, task_type)


def check_budget_remaining() -> dict:
    """Remaining daily budget; the circuit breaker's data source."""
    from tokentriage.config import load_policy
    cap = float(load_policy()["default"]["daily_budget_usd"])
    spent = db.spend_today_usd()
    return {"cap_usd": cap, "spent_usd": round(spent, 6),
            "remaining_usd": round(max(0.0, cap - spent), 6)}


def log_routing_decision(task_id: str, chosen_tier: str, reason: str, cost_usd: float) -> str:
    """Append-only decision log entry (dashboard + audit trail)."""
    # decisions table already written by orchestrator._log; this is the
    # MCP-visible confirmation hook. Kept separate so external MCP callers
    # can log their own routed decisions too.
    return f"logged:{task_id}:{chosen_tier}:{cost_usd:.6f} ({reason[:60]})"


def get_routing_stats(window_hours: float = 24.0) -> dict:
    """Aggregates for dashboards/reports: cost, savings, cache hits, tiers."""
    return db.stats(window_hours)


def quarantine_request(task_preview: str, reason: str) -> str:
    """Security gateway writes flagged requests here for audit."""
    db.quarantine(task_preview, reason)
    return "quarantined"
