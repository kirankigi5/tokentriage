"""Routing guardrails: taxonomy backstop, sensitive pinning, policy tier
overrides, and the budget circuit breaker. All pure/deterministic — no models."""
import pytest

from tokentriage.agents.triage import TriageVerdict, _backstop, _taxonomy_hint, is_sensitive
from tokentriage.agents.orchestrator import _candidate_tiers
from tokentriage.security import budget as breaker
from tokentriage.security.gateway import SecurityError


def _v(task_type, score=0.1):
    return TriageVerdict(score, task_type, 50, "llm")


# --- deterministic taxonomy backstop --------------------------------------
@pytest.mark.parametrize("task,expected", [
    ("Classify this ticket as billing or account", "classification"),
    ("Is this review positive or negative: broke fast", "classification"),
    ("Write a SQL query for top customers", "code_generation"),
    ("Write a Python decorator with backoff", "code_generation"),
    ("Calculate the monthly savings step by step", "multi_step_reasoning"),
    ("Summarize in two sentences: ...", "summarization"),
])
def test_taxonomy_backstop_corrects_obvious_mislabels(task, expected):
    # even if the LLM wrongly says factual_lookup, the backstop corrects it
    assert _backstop(task, _v("factual_lookup")).task_type == expected


def test_taxonomy_hint_none_for_plain_fact():
    assert _taxonomy_hint("What is the capital of Australia?") is None


# --- sensitive-domain pinning ---------------------------------------------
def test_sensitive_pins_to_legal_over_taxonomy_and_llm():
    # a tax question the LLM mislabelled classification must still pin to legal
    v = _backstop("Explain the tax implications of a contractor", _v("classification"))
    assert v.task_type == "legal_or_financial"


def test_is_sensitive_detection():
    assert is_sensitive("recognizing revenue on a contract")
    assert not is_sensitive("what is the capital of france")


# --- policy per-task-type tier overrides ----------------------------------
def test_policy_min_tier_forces_high_tier():
    pol = {"task_overrides": {"legal_or_financial": {"min_tier": "T3"}}}
    assert _candidate_tiers(pol, _v("legal_or_financial")) == ["T3"]


def test_policy_max_tier_caps_cheap():
    pol = {"task_overrides": {"factual_lookup": {"max_tier": "T1"}}}
    assert _candidate_tiers(pol, _v("factual_lookup")) == ["T1"]


def test_no_override_allows_full_ladder():
    assert _candidate_tiers({}, _v("summarization")) == ["T1", "T2", "T3"]


# --- budget circuit breaker -----------------------------------------------
def test_budget_breaker_disallows_expensive_when_capped(monkeypatch):
    monkeypatch.setattr(breaker.db, "spend_today_usd", lambda: 999.0)
    pol = {"default": {"daily_budget_usd": 5.0}}
    assert breaker.allowed_tiers(pol, ["T1", "T2", "T3"]) == ["T1"]


def test_budget_breaker_429_when_only_expensive_left(monkeypatch):
    monkeypatch.setattr(breaker.db, "spend_today_usd", lambda: 999.0)
    pol = {"default": {"daily_budget_usd": 5.0}}
    with pytest.raises(SecurityError):
        breaker.allowed_tiers(pol, ["T2", "T3"])


def test_budget_breaker_allows_all_under_cap(monkeypatch):
    monkeypatch.setattr(breaker.db, "spend_today_usd", lambda: 0.0)
    pol = {"default": {"daily_budget_usd": 5.0}}
    assert breaker.allowed_tiers(pol, ["T1", "T2", "T3"]) == ["T1", "T2", "T3"]
