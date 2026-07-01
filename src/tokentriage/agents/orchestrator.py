"""Orchestrator Agent (ADK) — owns the routing state machine.

Per-request lifecycle (each transition is logged with a timestamp, which
produces the single-request trace shown in the demo video):

  SANITIZED -> CACHED? -> TRIAGED -> PRICED -> POLICY_CHECKED
            -> DISPATCHED -> VERIFIED? -> (ESCALATED?) -> LOGGED

Decision function, stated plainly:
  choose the CHEAPEST tier whose benchmark accuracy for this task type
  >= the policy accuracy floor, subject to per-type min/max tier overrides
  and the remaining daily budget.

Pricing/benchmarks/budget are fetched through the MCP server tools rather
than read directly — any MCP-compatible agent could consume the same cost
intelligence, which is the interoperability MCP exists for.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from tokentriage import db, providers
from tokentriage.agents.triage import TriageVerdict, triage
from tokentriage.agents.verifier import should_sample, verify
from tokentriage.cache.semantic_cache import SemanticCache
from tokentriage.mcp_server import tools as mcp_tools
from tokentriage.models.registry import (
    TIERS, TIER_ORDER, estimate_baseline_usd, estimate_cost_usd, next_tier_up)
from tokentriage.security import budget as breaker


@dataclass
class RouteResult:
    task_id: str
    answer: str
    chosen_tier: str
    task_type: str
    complexity: float
    rationale: str
    cost_usd: float
    baseline_cost_usd: float
    cache_hit: bool = False
    verified: bool = False
    verdict: str | None = None
    escalated_to: str | None = None
    trace: list[tuple[str, float]] = field(default_factory=list)  # (state, ts)


# --- ADK agent definition -------------------------------------------------
# The orchestrator is exposed as an ADK Agent with triage/verifier as
# sub-agents and the MCP toolset attached, so the whole system is drivable
# from an ADK Runner (course requirement: multi-agent system in code).
# MCPToolset wired via stdio to the TokenTriage MCP server; ADK manages the
# connection lifecycle. The direct-call path below is the fallback that keeps
# the gateway functional regardless.
try:
    from google.adk.agents import Agent
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams
    from mcp.client.stdio import StdioServerParameters
    from tokentriage.agents.triage import triage_agent
    from tokentriage.agents.verifier import verifier_agent

    orchestrator_agent = Agent(
        name="tokentriage_orchestrator",
        model=TIERS["T2"].model_id,
        description="Routes tasks to the cheapest sufficient model tier.",
        instruction="Coordinate triage, policy-checked dispatch, and verification.",
        sub_agents=[a for a in (triage_agent, verifier_agent) if a],
        tools=[
            MCPToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="python",
                        args=["-m", "tokentriage.mcp_server.server"],
                    )
                )
            )
        ],
    )
except ImportError:
    orchestrator_agent = None


def _candidate_tiers(policy: dict, verdict: TriageVerdict) -> list[str]:
    """Apply per-task-type min/max tier overrides to the ordered tier list."""
    paid = [t for t in TIER_ORDER if t != "T0"]
    ov = policy.get("task_overrides", {}).get(verdict.task_type, {})
    lo = ov.get("min_tier")
    hi = ov.get("max_tier")
    if lo:
        paid = [t for t in paid if TIER_ORDER.index(t) >= TIER_ORDER.index(lo)]
    if hi:
        paid = [t for t in paid if TIER_ORDER.index(t) <= TIER_ORDER.index(hi)]
    return paid


def _pick_tier(policy: dict, verdict: TriageVerdict) -> tuple[str, str]:
    """Core decision: cheapest candidate clearing the accuracy floor."""
    floor = float(policy["default"]["accuracy_floor"])
    candidates = _candidate_tiers(policy, verdict)
    candidates = breaker.allowed_tiers(policy, candidates)  # budget circuit breaker
    for tier in candidates:  # ordered cheapest-first
        acc = mcp_tools.get_accuracy_benchmark(tier, verdict.task_type)
        if acc >= floor:
            return tier, f"cheapest tier with {acc:.2f} >= floor {floor:.2f} for {verdict.task_type}"
    # Nothing clears the floor within constraints: take the most capable allowed.
    return candidates[-1], "no tier met floor; selected most capable allowed tier"


def _dispatch(tier: str, task: str) -> tuple[str, int, int]:
    """Call the chosen model via the provider layer (Ollama/Gemini/OpenAI).
    Returns (answer, input_tokens, output_tokens) with real backend counts."""
    return providers.generate(TIERS[tier], task)


def route(task: str, policy: dict, cache: SemanticCache) -> RouteResult:
    """Full state machine for one request. Task is ALREADY gateway-sanitized."""
    task_id = uuid.uuid4().hex[:12]
    trace: list[tuple[str, float]] = [("SANITIZED", time.time())]

    # --- T0: semantic cache -------------------------------------------------
    if policy.get("cache", {}).get("enabled", True):
        hit = cache.lookup(task)
        if hit is not None:
            trace.append(("CACHE_HIT", time.time()))
            baseline = estimate_baseline_usd(len(task) // 4, len(hit) // 4)
            r = RouteResult(task_id, hit, "T0", "cached", 0.0,
                            "semantic cache hit", 0.0, baseline,
                            cache_hit=True, trace=trace)
            _log(r, task)
            return r

    # --- Triage ---------------------------------------------------------------
    verdict = triage(task)
    trace.append(("TRIAGED", time.time()))

    # --- Price + policy + budget ------------------------------------------
    tier, why = _pick_tier(policy, verdict)
    trace.append(("POLICY_CHECKED", time.time()))

    # --- Dispatch -------------------------------------------------------------
    answer, itok, otok = _dispatch(tier, task)
    cost = estimate_cost_usd(tier, itok, otok)
    breaker.record(tier, cost)
    baseline = estimate_baseline_usd(itok, otok)  # what all-cloud-frontier would cost
    trace.append(("DISPATCHED", time.time()))

    result = RouteResult(task_id, answer, tier, verdict.task_type,
                         verdict.complexity_score,
                         f"{verdict.rationale} | {why}", cost, baseline, trace=trace)

    # --- Sampled verification + bounded escalation -------------------------
    max_hops = int(policy.get("escalation", {}).get("max_hops", 2))
    hops = 0
    while should_sample(policy, result.chosen_tier) and hops < max_hops:
        result.verified = True
        vr = verify(task, result.answer)
        result.verdict = vr.verdict
        db.record_feedback(verdict.task_type, result.chosen_tier, vr.verdict)
        trace.append((f"VERIFIED_{vr.verdict.upper()}", time.time()))
        if vr.verdict == "pass":
            break
        up = next_tier_up(result.chosen_tier)
        if up is None:
            break
        # Escalate: re-answer one tier up; costs accumulate honestly.
        answer, itok, otok = _dispatch(up, task)
        ecost = estimate_cost_usd(up, itok, otok)
        breaker.record(up, ecost)
        result.answer = answer
        result.cost_usd += ecost
        result.escalated_to = up
        result.chosen_tier = up
        hops += 1
        trace.append((f"ESCALATED_{up}", time.time()))

    # --- Cache the final answer + log --------------------------------------
    if policy.get("cache", {}).get("enabled", True) and not result.cache_hit:
        cache.store(task, result.answer)
    _log(result, task)
    return result


def _log(r: RouteResult, task: str) -> None:
    db.log_decision(
        task_id=r.task_id, task_preview=task[:120], task_type=r.task_type,
        complexity=r.complexity, chosen_tier=r.chosen_tier, rationale=r.rationale,
        cost_usd=r.cost_usd, baseline_cost_usd=r.baseline_cost_usd,
        cache_hit=int(r.cache_hit), verified=int(r.verified),
        verdict=r.verdict, escalated_to=r.escalated_to,
    )
    mcp_tools.log_routing_decision(r.task_id, r.chosen_tier, r.rationale, r.cost_usd)
