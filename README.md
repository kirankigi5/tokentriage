# TokenTriage — Inference Cost Engine

> Route every task to the cheapest model that can still answer it correctly.

TokenTriage is a cost-aware routing layer for LLM-powered applications. It sits in
front of any agent pipeline as an OpenAI-compatible proxy, triages each incoming
task with a lightweight classifier agent, and dispatches it to the cheapest model
tier whose measured accuracy for that task type clears a configurable floor.
A verifier agent samples cheap-tier answers and escalates failures, and a
feedback store adapts routing thresholds over time — so the system doesn't just
route cheap, it catches its own mistakes and gets smarter with use.

**Capstone:** Kaggle × Google — AI Agents: Intensive Vibe Coding (Track: Agents for Business)

---

## The problem

Businesses running LLM-powered agents (support bots, ops assistants, copilots)
pay top-tier prices for every query — including the large fraction of tasks a
much cheaper model handles correctly. There is no intelligent layer deciding,
per task, which model is *sufficient*. TokenTriage is that layer.

## Architecture

```
client ──► FastAPI gateway (OpenAI-compatible)
              │
              ▼
        Security Gateway ── sanitizer · injection screen · rate limit · budget precheck
              │
              ▼
        Semantic Cache (T0, $0) ── hit? return
              │ miss
              ▼
        Orchestrator Agent (ADK)
          ├── Triage Agent (ADK): complexity score + task taxonomy + rationale
          ├── MCP Server: pricing · per-task benchmarks · budget · decision log
          └── Policy Engine (YAML): accuracy floor · tier overrides · caps
              │
              ▼
        Model Pool  T1 Flash-Lite → T2 Flash → T3 Pro
              │
              ▼
        Verifier Agent (ADK, sampled) ──fail──► Escalation (tier+1, re-verify)
              │
              ▼
        Feedback Store ──► `tokentriage tune` adapts thresholds
              │
              ▼
        Dashboard: spend vs baseline · savings % · cache hits · escalations
```

See `docs/architecture.md` for the full diagram and design rationale.

## Key concepts demonstrated (course rubric)

| Concept | Where |
|---|---|
| Multi-agent system (ADK) | `src/tokentriage/agents/` — Orchestrator, Triage, Verifier |
| MCP Server | `src/tokentriage/mcp_server/server.py` — 6 custom tools |
| Security features | `src/tokentriage/security/` — gateway, budget circuit breaker, key isolation |
| Deployability | OpenAI-compatible proxy (`proxy/app.py`), Dockerfile, Cloud Run notes |
| Agent skills (CLI) | `src/tokentriage/cli.py` — serve / benchmark / report / tune / attack-test |
| Antigravity | Built in the Antigravity IDE (see video) |

## Quickstart

```bash
# 1. Install (Python 3.11+)
pip install -e .

# 2. Configure — NEVER commit real keys
cp .env.example .env   # then fill in your keys

# 3. Run the gateway + dashboard
tokentriage serve      # http://localhost:8000  ·  /dashboard

# 4. Point any OpenAI-compatible client at it
#    base_url="http://localhost:8000/v1", model="tokentriage-auto"

# 5. Run the benchmark suite vs an always-Pro baseline
tokentriage benchmark
tokentriage report
```

## Configuration

Routing behavior is declarative — edit `config/policy.yaml`, not code:
accuracy floor, latency SLO, daily budget, per-task-type tier overrides
(e.g. `legal_or_financial: min_tier T3`), verification sample rate.

## Security notes

- All API keys via environment variables; per-tier key isolation.
- Input sanitizer + prompt-injection screen quarantine suspicious requests
  (see `tokentriage attack-test`).
- Budget circuit breaker halts expensive-tier routing at the daily cap.

## Repository layout

```
config/policy.yaml          declarative routing policy
src/tokentriage/
  agents/                   ADK agents: orchestrator, triage, verifier
  mcp_server/               custom MCP server (pricing, benchmarks, budget, logs)
  security/                 gateway + budget circuit breaker
  cache/                    semantic cache (tier 0, $0)
  models/                   model tier registry
  proxy/                    FastAPI OpenAI-compatible gateway + dashboard
  cli.py                    Typer CLI (agent skills)
  db.py                     SQLite persistence
benchmarks/                 30-query suite + always-Pro baseline
docs/architecture.md        design rationale
```

## Results

<!-- TODO: fill after running `tokentriage benchmark` -->
| Metric | Value |
|---|---|
| Total cost (TokenTriage) | $__ |
| Total cost (always-Pro baseline) | $__ |
| Savings | __% |
| Cache hit rate | __% |
| Verification escalations | __ |
| Accuracy vs floor | __ |

## License

MIT
