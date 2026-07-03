# TokenTriage — Architecture & Design Rationale

For the judge-facing visual map, run `tokentriage serve` and open
`http://localhost:8000/architecture`. The page includes clickable components
and scenario paths for cache hit, cheap route, sensitive escalation, and
security block.

## Decision function

> Choose the cheapest tier whose measured accuracy for this task type meets
> the policy accuracy floor, subject to per-type tier overrides, the latency
> SLO, and the remaining daily budget.

Simple to state; the engineering is in making each input to that function
trustworthy: triage supplies the task type, the MCP server supplies measured
per-type accuracy (learned, not assumed), the policy supplies the floor and
overrides, and the budget dashboard supplies the constraint.

## Request lifecycle (state machine)

SANITIZED -> CACHE? -> TRIAGED -> PRICED -> POLICY_CHECKED -> DISPATCHED
-> VERIFIED (sampled) -> ESCALATED (bounded) -> LOGGED

Every transition is timestamped in the decision log, so any single request
can be replayed as a trace (shown in the demo video).

## The three agents (ADK)

| Agent | Model | Job |
|---|---|---|
| Orchestrator | T2 | Owns the state machine; coordinates sub-agents + MCP tools |
| Triage | T1 (cheapest) | Structured verdict: complexity, task type, est. tokens, rationale |
| Verifier | T2 (mid) | Sampled pass/fail judgment on cheap-tier answers |

Meta-point: the router itself is cost-optimized — triage runs on the cheapest
tier; verification is sampled, not universal.

## Why MCP

Pricing, per-task-type benchmarks, budget, and stats live behind an MCP
server rather than being hardcoded. Any MCP-compatible agent — not just our
orchestrator — can consume TokenTriage's cost intelligence. Interoperability
is the point of the protocol; this design uses it for its actual purpose.

## Security model

Defense-in-depth, all pre-spend:
1. Sanitizer (length, control chars) — nothing malformed reaches a model.
2. Injection screen (auditable regex heuristics) — flagged input is
   quarantined and logged, never silently dropped. `tokentriage attack-test`
   demonstrates it on command.
3. Rate limiter (token bucket per client key).
4. Budget circuit breaker — expensive tiers shut off at the daily cap;
   the service degrades gracefully to the cheap tier instead of failing.
5. Per-tier API key isolation via separate env vars.
6. Policy `min_tier` for sensitive task types (legal/financial never
   downgraded) — a safety property expressed as configuration.

## The learning loop

Every sampled verification writes a (task_type, tier, verdict) tuple.
`tokentriage tune` blends learned pass-rates into the benchmark table
(confidence-weighted so a few unlucky verdicts can't whipsaw routing).
Repeated cheap-tier failures on a task type raise its effective difficulty,
and future tasks of that type route higher automatically — the system
provably improves with use.

## Honest limitations

- Seeded benchmark accuracies are estimates until tuned from real feedback.
- The breaker is check-then-record; a single in-flight request can slightly
  overshoot the daily cap (accepted at demo scale, documented here).
- Brute-force cosine cache is O(n); production path is sqlite-vec / a vector DB.
- Verifier judgments are themselves model outputs and can be wrong; sampling
  rate and max_hops bound the blast radius in both directions.
