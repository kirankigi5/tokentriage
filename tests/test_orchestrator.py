import pytest
from unittest.mock import patch, MagicMock

from tokentriage.agents.triage import TriageVerdict
from tokentriage.agents.orchestrator import (
    _candidate_tiers, _dispatch_with_failover, _next_escalation_tier, _pick_tier,
    _preferred_cloud_order, _preferred_cloud_tier)
from tokentriage.models.registry import ModelTier

def test_candidate_tiers_are_local_first():
    """First-pass routing stays local; cloud tiers are verifier rescue only."""
    
    # Mock TIERS with a mix of local and cloud models
    mock_tiers = {
        "T0": ModelTier("T0", "cache", 0.0, 0.0, "", "cache"),
        "T1": ModelTier("T1", "local", 0.0, 0.0, "", "ollama"),
        "T2": ModelTier("T2", "cloud1", 0.0, 0.0, "KEY_T2", "gemini"),  # Cloud, missing key
        "T3": ModelTier("T3", "cloud2", 0.0, 0.0, "KEY_T3", "openai"),  # Cloud, has key
    }
    
    with patch("tokentriage.models.registry.TIERS", mock_tiers):
        with patch("tokentriage.agents.orchestrator.TIERS", mock_tiers):
            with patch("tokentriage.config.tier_key", side_effect=lambda env, provider=None: "secret" if env == "KEY_T3" else ""):
                with patch("tokentriage.agents.orchestrator.TIER_ORDER", ["T0", "T1", "T2", "T3"]):
                    verdict = TriageVerdict(task_type="factual_lookup", complexity_score=1.0, estimated_output_tokens=100, rationale="")
                    policy = {"task_overrides": {}}
                    
                    candidates = _candidate_tiers(policy, verdict)
                    
                    assert candidates == ["T1"]


def test_escalation_can_choose_available_cloud_rescue():
    mock_tiers = {
        "T0": ModelTier("T0", "cache", 0.0, 0.0, "", "cache"),
        "T1": ModelTier("T1", "local", 0.0, 0.0, "", "ollama"),
        "T2": ModelTier("T2", "local2", 0.0, 0.0, "", "ollama"),
        "T3": ModelTier("T3", "local3", 0.0, 0.0, "", "ollama"),
        "T4": ModelTier("T4", "gemma", 0.0, 0.0, "KEY_T4", "openrouter"),
        "T5": ModelTier("T5", "gpt-oss", 0.0, 0.0, "KEY_T5", "openrouter"),
        "T6": ModelTier("T6", "qwen-next", 0.0, 0.0, "KEY_T6", "openrouter"),
        "T7": ModelTier("T7", "qwen-coder", 0.0, 0.0, "KEY_T7", "openrouter"),
    }

    with patch("tokentriage.agents.orchestrator.TIERS", mock_tiers):
        with patch("tokentriage.config.tier_key", side_effect=lambda env, provider=None: "secret" if env == "KEY_T7" else ""):
            with patch("tokentriage.agents.orchestrator.TIER_ORDER", ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]):
                assert _next_escalation_tier("T2", {"default": {"daily_budget_usd": 5.0}}, "code_generation") == "T7"


def test_preferred_cloud_tier_matches_task_type():
    assert _preferred_cloud_tier("code_generation") == "T7"
    assert _preferred_cloud_tier("multi_step_reasoning") == "T6"
    assert _preferred_cloud_tier("factual_lookup") == "T5"
    assert _preferred_cloud_tier("legal_or_financial") == "T4"
    assert _preferred_cloud_order("legal_or_financial") == ["T4", "T6", "T7", "T5"]


def test_prefer_cloud_override_routes_sensitive_to_openrouter():
    mock_tiers = {
        "T0": ModelTier("T0", "cache", 0.0, 0.0, "", "cache"),
        "T1": ModelTier("T1", "local", 0.0, 0.0, "", "ollama"),
        "T2": ModelTier("T2", "local2", 0.0, 0.0, "", "ollama"),
        "T3": ModelTier("T3", "local3", 0.0, 0.0, "", "ollama"),
        "T4": ModelTier("T4", "gemma", 0.0, 0.0, "KEY_T4", "openrouter"),
    }

    with patch("tokentriage.agents.orchestrator.TIERS", mock_tiers):
        with patch("tokentriage.agents.orchestrator.TIER_ORDER", ["T0", "T1", "T2", "T3", "T4"]):
            with patch("tokentriage.config.tier_key", side_effect=lambda env, provider=None: "secret" if env == "KEY_T4" else ""):
                tier, why = _pick_tier(
                    {"default": {"accuracy_floor": 0.90, "daily_budget_usd": 5.0},
                     "task_overrides": {"legal_or_financial": {"min_tier": "T3", "prefer_cloud": True}}},
                    TriageVerdict(0.8, "legal_or_financial", 100, "sensitive"),
                )

    assert tier == "T4"
    assert "OpenRouter" in why


def test_dispatch_failover_does_not_treat_provider_error_as_answer():
    events = []
    mock_tiers = {
        "T0": ModelTier("T0", "cache", 0.0, 0.0, "", "cache"),
        "T4": ModelTier("T4", "gemma", 0.0, 0.0, "KEY_T4", "openrouter"),
        "T5": ModelTier("T5", "gpt-oss", 0.0, 0.0, "KEY_T5", "openrouter"),
        "T6": ModelTier("T6", "qwen-next", 0.0, 0.0, "KEY_T6", "openrouter"),
    }

    def fake_dispatch(tier, task, messages):
        if tier == "T4":
            return "", 0, 0, 10.0, "RateLimitError: 429"
        return "rescued answer", 7, 9, 20.0, None

    with patch("tokentriage.agents.orchestrator.TIERS", mock_tiers):
        with patch("tokentriage.agents.orchestrator.TIER_ORDER", ["T0", "T4", "T5", "T6"]):
            with patch("tokentriage.config.tier_key", return_value="secret"):
                with patch("tokentriage.agents.orchestrator._timed_dispatch", side_effect=fake_dispatch):
                    answer, itok, otok, latency, err, final_tier, _ = _dispatch_with_failover(
                        "T4", "task", [{"role": "user", "content": "task"}],
                        {"default": {"daily_budget_usd": 5.0}},
                        TriageVerdict(0.8, "legal_or_financial", 100, "sensitive"),
                        lambda stage, **detail: events.append((stage, detail)),
                    )

    assert answer == "rescued answer"
    assert (itok, otok) == (7, 9)
    assert latency == 30.0
    assert err is None
    assert final_tier == "T6"
    assert ("ERROR", {"detail": "gemma failed: RateLimitError: 429"}) in events
    assert not any(stage == "DISPATCHED" and detail.get("detail") == "answered by gemma"
                   for stage, detail in events)
