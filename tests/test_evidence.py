from dataclasses import dataclass, field

from tokentriage import db
from tokentriage import evidence as ev


@dataclass
class DummyVerdict:
    complexity_score: float = 0.1
    task_type: str = "factual_lookup"
    confidence: int = 90
    rationale: str = "test"


@dataclass
class DummyResult:
    task_id: str = "dummy"
    answer: str = "ok"
    chosen_tier: str = "T1"
    task_type: str = "factual_lookup"
    complexity: float = 0.1
    rationale: str = "test"
    cost_usd: float = 0.000001
    baseline_cost_usd: float = 0.0001
    cache_hit: bool = False
    verified: bool = False
    verdict: str | None = None
    escalated_to: str | None = None
    context_note: str | None = None
    trace: list = field(default_factory=list)


def test_run_evidence_writes_expected_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "settings", type("S", (), {"db_path": str(tmp_path / "t.db")})())
    monkeypatch.setattr(ev, "load_policy", lambda: {
        "default": {"accuracy_floor": 0.9, "daily_budget_usd": 5.0},
        "cache": {"enabled": False},
    })
    monkeypatch.setattr(ev, "triage", lambda task: DummyVerdict())
    monkeypatch.setattr(ev, "route", lambda task, policy, cache: DummyResult())

    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"task":"What is the capital of Australia?","expect_type":"factual_lookup"}\n')

    run_dir = ev.run_evidence(queries, tmp_path / "reports")

    assert (run_dir / "report.md").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "routing_results.csv").exists()
    assert (run_dir / "dashboard.html").exists()
    assert (tmp_path / "reports" / "latest" / "dashboard.html").exists()


def test_chainlit_demo_imports_without_running_server():
    import demo_chainlit

    assert demo_chainlit.SCENARIOS
