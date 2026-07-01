# TokenTriage — Inference Cost Engine

> Route every task to the cheapest model that can still answer it correctly —
> and run the cheap tiers on **free local open-source models**.

TokenTriage is a cost-aware routing layer for LLM-powered applications. It sits
in front of any agent pipeline as an OpenAI-compatible proxy, triages each
incoming task with a lightweight classifier agent, and dispatches it to the
cheapest model tier whose measured accuracy for that task type clears a
configurable floor. A verifier agent samples cheap-tier answers and escalates
failures, and a feedback store adapts routing thresholds over time — so the
system doesn't just route cheap, it catches its own mistakes and improves with
use.

The tiers are **hybrid**: cheap tasks run on local open-source models (Ollama)
for ~$0, and only genuinely hard tasks need a bigger model. Savings are measured
against the real business alternative — **sending every task to a frontier cloud
model**.

**Capstone:** Kaggle × Google — AI Agents: Intensive Vibe Coding (Track: Agents for Business)

---

## The problem

Businesses running LLM-powered agents (support bots, ops assistants, copilots)
pay frontier-cloud prices for *every* query — including the large fraction of
tasks a much cheaper model handles correctly. There is no intelligent layer
deciding, per task, which model is *sufficient*. TokenTriage is that layer.

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
        Orchestrator Agent ── owns the routing state machine
          ├── Triage Agent: complexity score + task taxonomy + rationale
          ├── MCP Server: pricing · per-task benchmarks · budget · decision log
          └── Policy Engine (YAML): accuracy floor · tier overrides · caps
              │
              ▼
        Model Pool (via provider abstraction)
          T1 qwen2.5:3b  →  T2 qwen2.5:7b  →  T3 qwen2.5:14b     (local, Ollama, $0)
          (swap any tier for gemini / openai by config — see providers.py)
              │
              ▼
        Verifier Agent (sampled) ──fail──► Escalation (tier+1, re-verify)
              │
              ▼
        Feedback Store ──► `tokentriage tune` adapts thresholds
              │
              ▼
        Dashboard: spend vs all-cloud baseline · savings % · cache hits · escalations
```

Savings baseline = **all-cloud-frontier**: what every task *would* cost sent to
a top cloud model (gemini-2.5-pro list price). It is never called — it's the
yardstick. See `docs/architecture.md` for full design rationale.

## Why local + open-source

- **Real $0 cheap tiers.** T1/T2/T3 run on your machine via Ollama — no keys,
  no per-token bill, fully offline.
- **Provider abstraction** (`src/tokentriage/providers.py`): every tier names a
  provider (`ollama` | `gemini` | `openai`). Adding a paid cloud tier is a
  registry edit — nothing above the provider layer changes.
- **Phased:** Phase 1 = all local. Phase 2 = add free-tier Gemini as a tier.
  Phase 3 = add paid OpenAI / Google / Mistral. Same code.

## Key concepts demonstrated (course rubric)

| Concept | Where |
|---|---|
| Multi-agent system (Google ADK) | Triage + Verifier are ADK `LlmAgent`s running on local Ollama via LiteLLM (`TOKENTRIAGE_USE_ADK=1`; see `tokentriage adk-demo`), coordinated by a deterministic orchestrator |
| MCP Server | `src/tokentriage/mcp_server/server.py` — custom pricing/benchmark/budget/log tools |
| Security features | `src/tokentriage/security/` — gateway, injection screen, budget circuit breaker, key isolation, sensitive-task backstop |
| Deployability | OpenAI-compatible proxy (`proxy/app.py`), Dockerfile |
| Agent skills (CLI) | `src/tokentriage/cli.py` — serve / benchmark / report / tune / adk-demo / attack-test |
| Antigravity | Built in the Antigravity IDE (see video) |

**ADK note:** routing/cost/escalation is deterministic *by design* — you don't
want an LLM guessing price tiers. The genuine LLM agents are Triage and
Verifier; both run through the ADK Runner on local models. `tokentriage
adk-demo "<task>"` shows them executing.

## Quickstart (fully local, no API keys)

```bash
# 1. Install Ollama (https://ollama.com) and pull the model tiers
ollama pull qwen2.5:3b qwen2.5:7b qwen2.5:14b nomic-embed-text

# 2. Install TokenTriage (Python 3.11+)
pip install -e .

# 3. (optional) configure — defaults already run fully local
cp .env.example .env

# 4. Run the gateway + dashboard
tokentriage serve            # http://localhost:8000  ·  /dashboard

# 5. Point any OpenAI-compatible client at it
#    base_url="http://localhost:8000/v1"   (the `model` field is ignored —
#    TokenTriage picks the tier)

# 6. Run the benchmark suite vs the all-cloud baseline
tokentriage benchmark
tokentriage report
```

To add a cloud tier later: edit a tier in `src/tokentriage/models/registry.py`
(set `provider="gemini"`, a model id, and its price), then put a key in `.env`
(`GEMINI_API_KEY=...`). One shared key powers any cloud tier; per-tier vars
override it for key isolation.

## Configuration

Routing behavior is declarative — edit `config/policy.yaml`, not code:
accuracy floor, latency SLO, daily budget, per-task-type tier overrides
(e.g. `legal_or_financial: min_tier T3`), verification sample rate.

## Security notes

- No keys needed for local operation; cloud keys (if used) via environment
  variables only, with per-tier key isolation.
- Input sanitizer + prompt-injection screen quarantine suspicious requests
  (see `tokentriage attack-test`).
- Budget circuit breaker halts expensive-tier routing at the daily cap.

## Repository layout

```
config/policy.yaml          declarative routing policy
src/tokentriage/
  agents/                   orchestrator, triage, verifier
  providers.py              provider abstraction (ollama / gemini / openai)
  mcp_server/               custom MCP server (pricing, benchmarks, budget, logs)
  security/                 gateway + budget circuit breaker
  cache/                    semantic cache (tier 0, $0) — local embeddings
  models/registry.py        model tier registry + cloud baseline reference
  proxy/                    FastAPI OpenAI-compatible gateway + dashboard
  cli.py                    Typer CLI (agent skills)
  db.py                     SQLite persistence
benchmarks/                 30-query suite + baseline
docs/architecture.md        design rationale
```

## Results

Real run: 30-query business workload, all tiers on local open-source models
(Apple M4), measured against the all-cloud-frontier baseline.

| Metric | Value |
|---|---|
| Total cost (TokenTriage, local) | **$0.00172** |
| Total cost (all-cloud-frontier baseline) | $0.08892 |
| **Savings** | **98.1%** |
| Cache hits ($0) | 3 / 30 |
| Verification escalations | 1 |
| Tier utilization (T0/T1/T2/T3) | 3 / 13 / 5 / 9 |

*Baseline = every task sent to gemini-2.5-pro (list price). TokenTriage runs
all tiers on local open-source models, escalating only when the verifier flags
a weak answer. Numbers reproduce via `tokentriage benchmark` (Apple M4).*

## License

MIT
