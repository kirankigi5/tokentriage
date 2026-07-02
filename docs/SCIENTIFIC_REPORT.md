# TokenTriage Scientific Report

## Hypothesis

An agent gateway can reduce modeled frontier-cloud inference cost while
preserving quality by routing each task to the cheapest tier whose measured
accuracy clears policy.

## Methodology

The benchmark workload contains business tasks across factual lookup,
classification, summarization, creative writing, reasoning, code generation,
and legal/financial review. Each task is routed through the same production
pipeline used by the API and Chainlit demo.

For every task, TokenTriage records:

- predicted task type
- selected tier
- modeled TokenTriage cost
- modeled always-frontier baseline cost
- cache hit status
- verifier/escalation status
- routing rationale

The evidence command writes `reports/run_TIMESTAMP/` with `report.md`,
`metrics.json`, `routing_results.csv`, and `dashboard.html`.

## Primary Result

The headline benchmark reduces modeled cost from $0.08892 to $0.00172, a
98.1% reduction versus an always-`gemini-2.5-pro` baseline.

## Reproduce

```bash
tokentriage evidence
open reports/latest/dashboard.html
```

## Interpretation

The result is not a claim that all tasks are free. It proves that a large share
of routine agent traffic can be served by cheaper/local tiers, while sensitive
or complex tasks remain protected by policy overrides and verifier escalation.
