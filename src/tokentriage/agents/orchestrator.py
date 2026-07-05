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
    TIERS, TIER_ORDER, estimate_baseline_usd, estimate_cost_usd)
from tokentriage.security import budget as breaker
from tokentriage.security.gateway import SecurityError


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
    context_note: str | None = None  # how conversation context was shared (cloud)
    dispatch_latency_ms: float = 0.0  # total provider dispatch latency, incl. escalation
    error: str | None = None
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
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams
    from mcp.client.stdio import StdioServerParameters
    from tokentriage.agents.triage import triage_agent
    from tokentriage.agents.verifier import verifier_agent

    orchestrator_agent = Agent(
        name="tokentriage_orchestrator",
        model=LiteLlm(model=f"ollama_chat/{TIERS['T2'].model_id}"),
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
    # First-pass routing is local-first by design. Cloud/OpenRouter tiers are
    # quality rescue targets, entered only after verifier failure.
    paid = [
        t for t in TIER_ORDER
        if t != "T0" and TIERS[t].provider not in ("gemini", "openai", "openrouter")
    ]
    ov = policy.get("task_overrides", {}).get(verdict.task_type, {})
    lo = ov.get("min_tier")
    hi = ov.get("max_tier")
    if lo:
        paid = [t for t in paid if TIER_ORDER.index(t) >= TIER_ORDER.index(lo)]
    if hi:
        paid = [t for t in paid if TIER_ORDER.index(t) <= TIER_ORDER.index(hi)]
    valid_paid = []
    for t in paid:
        tier_obj = TIERS[t]
        if tier_obj.provider in ("gemini", "openai", "openrouter") and not tier_obj.api_key:
            continue
        valid_paid.append(t)
    return valid_paid


def _available_tier(tier: str) -> bool:
    tier_obj = TIERS[tier]
    return not (tier_obj.provider in ("gemini", "openai", "openrouter") and not tier_obj.api_key)


def _preferred_cloud_tier(task_type: str) -> str | None:
    """Pick the OpenRouter rescue model that best matches the failed task."""
    order = _preferred_cloud_order(task_type)
    return order[0] if order else None


def _preferred_cloud_order(task_type: str) -> list[str]:
    """Rank OpenRouter rescue models by task type.

    The first model can be rate-limited upstream, so failover should move to
    the next sensible model rather than simply the next tier number.
    """
    return {
        "code_generation": "T7",          # qwen/qwen3-coder-480b-a35b:free
        "multi_step_reasoning": "T6",     # qwen/qwen3-next-80b-a3b-instruct:free
        "summarization": "T4,T6,T5",      # Gemma, then stronger reasoning, then fast fallback
        "legal_or_financial": "T4,T6,T7,T5",
        "creative_short": "T5",           # openai/gpt-oss-20b:free
        "classification": "T5",
        "factual_lookup": "T5",
    }.get(task_type, "").split(",") if task_type else []


def _next_escalation_tier(current: str, policy: dict, task_type: str = "") -> str | None:
    """Quality rescue prefers the first available cloud tier after local failure.

    If OpenRouter is configured, a failed local answer jumps there instead of
    burning time through every larger local model. Without a key, the no-key
    local ladder still works.
    """
    later = TIER_ORDER[TIER_ORDER.index(current) + 1:]
    later = [t for t in later if t != "T0" and _available_tier(t)]
    try:
        later = breaker.allowed_tiers(policy, later) if later else []
    except SecurityError:
        return None
    cloud = [t for t in later if TIERS[t].provider in ("gemini", "openai", "openrouter")]
    for preferred in _preferred_cloud_order(task_type):
        if preferred in cloud:
            return preferred
    return (cloud[0] if cloud else later[0]) if later else None


def _pick_tier(policy: dict, verdict: TriageVerdict) -> tuple[str, str]:
    """Core decision: cheapest candidate clearing the accuracy floor."""
    floor = float(policy.get("default", {}).get("accuracy_floor", 0.90))
    slo = float(policy.get("default", {}).get("latency_slo_ms", 4000))
    override = policy.get("task_overrides", {}).get(verdict.task_type, {})
    if override.get("prefer_cloud"):
        preferred = _preferred_cloud_tier(verdict.task_type)
        if preferred and _available_tier(preferred):
            try:
                allowed = breaker.allowed_tiers(policy, [preferred])
            except SecurityError:
                allowed = []
            if preferred in allowed:
                return preferred, (
                    f"policy prefers OpenRouter rescue tier {preferred} "
                    f"for {verdict.task_type}; local fallback remains available"
                )

    candidates = _candidate_tiers(policy, verdict)
    candidates = breaker.allowed_tiers(policy, candidates)  # budget circuit breaker
    
    fallback = None
    for tier in candidates:  # ordered cheapest-first
        acc = mcp_tools.get_accuracy_benchmark(tier, verdict.task_type)
        if acc >= floor:
            lat = mcp_tools.get_latency_benchmark(tier).get("p95_dispatch_latency_ms", 0.0)
            if lat > 0 and lat > slo:
                if fallback is None:
                    fallback = (tier, f"met accuracy but skipped due to latency (p95 {lat:.0f}ms > {slo}ms); used as fallback")
                continue
            return tier, f"cheapest tier with {acc:.2f} >= floor {floor:.2f} (p95 {lat:.0f}ms <= SLO) for {verdict.task_type}"
    
    if fallback:
        return fallback

    if not candidates:
        # Extreme edge case: all tiers disabled (missing keys, out of budget, etc.)
        raise RuntimeError("No candidate tiers available. Check API keys and budget.")

    # Nothing clears the floor within constraints: take the most capable allowed.
    return candidates[-1], "no tier met floor; selected most capable allowed tier"


def _privacy_context(model_tier, task: str, messages: list[dict] | None,
                     policy: dict) -> tuple[list[dict] | None, str | None]:
    """Decide what conversation context is sent to the chosen tier.

    Local tiers get the FULL history — it never leaves the machine. For CLOUD
    tiers, the privacy policy governs what's shared (full/none/last_n/summary),
    and a sensitive-content firewall strips any prior turn flagged
    legal/financial/medical so it can never reach a third party.
    Returns (messages_to_send, human-readable note).
    """
    from tokentriage.agents.triage import is_sensitive

    if messages is None:
        return None, None
    if model_tier.provider in ("ollama", "cache"):
        return messages, None  # on-device: full context, nothing leaves

    pol = policy.get("privacy", {})
    mode = pol.get("cloud_context", "full")
    current = [{"role": "user", "content": task}]  # sanitized latest turn
    prior = messages[:-1] if messages and messages[-1].get("role") == "user" else list(messages)

    firewalled = False
    if pol.get("sensitive_firewall", True):
        kept = [m for m in prior if not is_sensitive(m.get("content", ""))]
        firewalled = len(kept) != len(prior)
        prior = kept

    fw = " +firewall" if firewalled else ""
    if mode == "none":
        return current, f"cloud_context=none{fw}"
    if mode == "last_n":
        n = int(pol.get("context_last_n", 2))
        return prior[-(2 * n):] + current, f"cloud_context=last_{n}{fw}"
    if mode == "summary" and prior:
        joined = "\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in prior)
        summary, _, _ = providers.generate(
            TIERS["T1"],  # summarize LOCALLY so raw prior turns never leave
            f"Summarize this conversation in 2-3 sentences for context:\n{joined}")
        ctx = [{"role": "user", "content": f"[Prior context, summarized locally]: {summary}"}]
        return ctx + current, f"cloud_context=summary (local){fw}"
    return prior + current, (f"cloud_context=full{fw}" if fw else None)


def _dispatch(tier: str, task: str, messages: list[dict] | None = None) -> tuple[str, int, int]:
    """Call the chosen model via the provider layer (Ollama/Gemini/OpenAI).
    Returns (answer, input_tokens, output_tokens) with real backend counts.
    `messages` carries multi-turn context; falls back to the single task."""
    return providers.generate(TIERS[tier], task, messages)


def _timed_dispatch(tier: str, task: str,
                    messages: list[dict] | None = None) -> tuple[str, int, int, float, str | None]:
    """Dispatch and return latency in milliseconds and error string for routing telemetry."""
    start = time.perf_counter()
    error_msg = None
    answer, itok, otok = "", 0, 0
    try:
        answer, itok, otok = _dispatch(tier, task, messages)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
    latency_ms = (time.perf_counter() - start) * 1000
    return answer, itok, otok, latency_ms, error_msg


def _dispatch_with_failover(tier: str, task: str, messages: list[dict] | None,
                            policy: dict, verdict: TriageVerdict, emit) -> tuple[str, int, int, float, str | None, str, str | None]:
    """Dispatch, failing over to later rescue tiers on provider errors.

    Returns answer/tokens/latency/error/final_tier/context_note. A provider
    error is never treated as an answered response.
    """
    current = tier
    total_latency = 0.0
    last_error = None
    context_note = None
    tried = set()

    while current and current not in tried:
        tried.add(current)
        send_msgs, cnote = _privacy_context(TIERS[current], task, messages, policy)
        emit("DISPATCHING", tier=current, model=TIERS[current].model_id,
             detail=f"generating on {TIERS[current].model_id}")
        answer, itok, otok, latency_ms, err = _timed_dispatch(current, task, send_msgs)
        total_latency += latency_ms
        if cnote:
            context_note = cnote
            emit("CONTEXT", detail=cnote)
        if not err:
            emit("DISPATCHED", cost=estimate_cost_usd(current, itok, otok),
                 tokens=itok + otok, latency_ms=round(latency_ms, 1),
                 detail=f"answered by {TIERS[current].model_id}")
            return answer, itok, otok, total_latency, None, current, context_note

        last_error = err
        emit("ERROR", detail=f"{TIERS[current].model_id} failed: {err}")
        up = _next_escalation_tier(current, policy, verdict.task_type)
        if up is None or up in tried:
            break
        emit("ESCALATING", tier=up, model=TIERS[up].model_id,
             detail=f"{TIERS[current].model_id} failed — retrying on {TIERS[up].model_id}")
        current = up

    return "", 0, 0, total_latency, last_error, current or tier, context_note


def route(task: str, policy: dict, cache: SemanticCache,
          messages: list[dict] | None = None, on_event=None) -> RouteResult:
    """Full state machine for one request. Task is ALREADY gateway-sanitized.

    `task` (the latest user turn) drives cache/triage/routing/verify; `messages`
    (full conversation) is what the chosen model actually answers, so follow-ups
    keep context while routing still triages the current turn.
    """
    task_id = uuid.uuid4().hex[:12]
    trace: list[tuple[str, float]] = []

    def emit(stage: str, **detail):
        """Record a pipeline stage AND stream it live to any listener."""
        trace.append((stage, time.time()))
        if on_event:
            on_event(stage, detail)

    emit("SANITIZED", detail="input checked & sanitized")

    # --- T0: semantic cache -------------------------------------------------
    if policy.get("cache", {}).get("enabled", True):
        hit = cache.lookup(task)
        if hit is not None:
            emit("CACHE_HIT", detail="semantic match found — $0, no model call")
            baseline = estimate_baseline_usd(len(task) // 4, len(hit) // 4)
            r = RouteResult(task_id, hit, "T0", "cached", 0.0,
                            "semantic cache hit", 0.0, baseline,
                            cache_hit=True, trace=trace)
            emit("DONE", tier="T0", cost=0.0)
            _log(r, task)
            return r
        emit("CACHE_MISS", detail="no semantic match — routing")

    # --- Triage ---------------------------------------------------------------
    verdict = triage(task)
    emit("TRIAGED", task_type=verdict.task_type, complexity=verdict.complexity_score,
         detail=verdict.rationale)

    # --- Price + policy + budget ------------------------------------------
    tier, why = _pick_tier(policy, verdict)
    emit("ROUTED", tier=tier, model=TIERS[tier].model_id, detail=why)

    # --- Dispatch (privacy policy governs cloud context) ------------------
    answer, itok, otok, latency_ms, err, final_tier, cnote = _dispatch_with_failover(
        tier, task, messages, policy, verdict, emit)
    if final_tier != tier:
        emit(f"ESCALATED_{final_tier}", tier=final_tier, latency_ms=round(latency_ms, 1))
    tier = final_tier
    cost = estimate_cost_usd(tier, itok, otok)
    breaker.record(tier, cost)
    baseline = estimate_baseline_usd(itok, otok)  # what all-cloud-frontier would cost

    result = RouteResult(task_id, answer, tier, verdict.task_type,
                         verdict.complexity_score,
                         f"{verdict.rationale} | {why}", cost, baseline,
                         context_note=cnote, dispatch_latency_ms=latency_ms,
                         error=err, trace=trace)

    # --- Sampled verification + bounded escalation -------------------------
    max_hops = int(policy.get("escalation", {}).get("max_hops", 2))
    hops = 0
    while should_sample(policy, result.chosen_tier, task) and hops < max_hops:
        result.verified = True
        emit("VERIFYING", detail=f"sampling {result.chosen_tier} answer for quality")
        vr = verify(task, result.answer)
        result.verdict = vr.verdict
        db.record_feedback(verdict.task_type, result.chosen_tier, vr.verdict)
        emit(f"VERIFIED_{vr.verdict.upper()}", detail=vr.reason)
        if vr.verdict == "pass":
            break
        up = _next_escalation_tier(result.chosen_tier, policy, verdict.task_type)
        if up is None:
            break
        # Escalate: re-answer one tier up; costs accumulate honestly.
        emit("ESCALATING", tier=up, model=TIERS[up].model_id,
             detail=f"answer too thin — retrying on {TIERS[up].model_id}")
        send_msgs, cnote = _privacy_context(TIERS[up], task, messages, policy)
        answer, itok, otok, latency_ms, err = _timed_dispatch(up, task, send_msgs)
        ecost = estimate_cost_usd(up, itok, otok)
        breaker.record(up, ecost)
        if err:
            result.error = err
            emit("ERROR", detail=f"{TIERS[up].model_id} failed on escalation: {err}")
        else:
            result.answer = answer
        result.cost_usd += ecost
        result.dispatch_latency_ms += latency_ms
        result.escalated_to = up
        result.chosen_tier = up
        if cnote:
            result.context_note = cnote
        hops += 1
        emit(f"ESCALATED_{up}", tier=up, latency_ms=round(latency_ms, 1))

    # --- Cache the final answer + log --------------------------------------
    if policy.get("cache", {}).get("enabled", True) and not result.cache_hit:
        cache.store(task, result.answer)
    emit("DONE", tier=result.chosen_tier, cost=result.cost_usd)
    _log(result, task)
    return result


def _log(r: RouteResult, task: str) -> None:
    db.log_decision(
        task_id=r.task_id, task_preview=task[:120], task_type=r.task_type,
        complexity=r.complexity, chosen_tier=r.chosen_tier, rationale=r.rationale,
        cost_usd=r.cost_usd, baseline_cost_usd=r.baseline_cost_usd,
        cache_hit=int(r.cache_hit), verified=int(r.verified),
        verdict=r.verdict, escalated_to=r.escalated_to,
        dispatch_latency_ms=round(r.dispatch_latency_ms, 1),
        error=r.error,
    )
    mcp_tools.log_routing_decision(r.task_id, r.chosen_tier, r.rationale, r.cost_usd)
