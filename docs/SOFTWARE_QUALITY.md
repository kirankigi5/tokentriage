# TokenTriage Software Quality

## Architecture

TokenTriage separates gateway compatibility, routing policy, provider dispatch,
MCP cost intelligence, security checks, and persistence. The orchestrator owns
the routing state machine while model providers remain swappable behind one
`generate()` interface.

## Production Controls

- OpenAI-compatible and Gemini-compatible ingress.
- Policy YAML for accuracy floor, budget cap, tier overrides, cache, and privacy.
- Semantic cache as tier zero.
- Budget circuit breaker before expensive routing.
- Prompt-injection quarantine before model calls.
- Sensitive prior-turn firewall for cloud contexts.
- SQLite audit trail for decisions, feedback, spend, cache, and conversations.

## Test Coverage

The test suite covers deterministic taxonomy backstops, sensitive-task tier
pinning, budget breaker behavior, prompt-injection quarantine, rate limiting,
and cloud-context privacy.

## Evidence Discipline

`tokentriage evidence` creates machine-readable and human-readable artifacts
from the same routing path used by production endpoints. This keeps demo claims
grounded in repeatable runs rather than screenshots.
