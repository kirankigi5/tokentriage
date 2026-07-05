# TokenTriage Evidence Report

## Killer Metric

TokenTriage reduced modeled always-frontier inference cost by **97.7%** on this business workload.

| Metric | Value |
|---|---:|
| Requests | 30 |
| TokenTriage cost | $0.001962 |
| Always-frontier baseline | $0.083959 |
| Savings | 97.7% |
| Cache hits | 0 |
| Verifier escalations | 1 |
| Avg dispatch latency | 27371.4 ms |
| P95 dispatch latency | 77778.5 ms |
| Taxonomy accuracy | 100.0% |
| Security blocks | 3 |

## Artifacts

- `dashboard.html` - offline interactive evidence dashboard
- `metrics.json` - machine-readable metrics
- `routing_results.csv` - per-task routing ledger
