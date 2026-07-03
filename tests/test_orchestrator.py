import pytest
from unittest.mock import patch, MagicMock

from tokentriage.agents.triage import TriageVerdict
from tokentriage.agents.orchestrator import _candidate_tiers
from tokentriage.models.registry import ModelTier

def test_candidate_tiers_filters_missing_keys():
    """Verify that cloud tiers without API keys are excluded from candidates."""
    
    # Mock TIERS with a mix of local and cloud models
    mock_tiers = {
        "T0": ModelTier("T0", "cache", 0.0, 0.0, "", "cache"),
        "T1": ModelTier("T1", "local", 0.0, 0.0, "", "ollama"),
        "T2": ModelTier("T2", "cloud1", 0.0, 0.0, "KEY_T2", "gemini"),  # Cloud, missing key
        "T3": ModelTier("T3", "cloud2", 0.0, 0.0, "KEY_T3", "openai"),  # Cloud, has key
    }
    
    with patch("tokentriage.models.registry.TIERS", mock_tiers):
        with patch("tokentriage.agents.orchestrator.TIERS", mock_tiers):
            with patch("tokentriage.config.tier_key", side_effect=lambda env: "secret" if env == "KEY_T3" else ""):
                with patch("tokentriage.agents.orchestrator.TIER_ORDER", ["T0", "T1", "T2", "T3"]):
                    verdict = TriageVerdict(task_type="factual_lookup", complexity_score=1.0, estimated_output_tokens=100, rationale="")
                    policy = {"task_overrides": {}}
                    
                    candidates = _candidate_tiers(policy, verdict)
                    
                    assert candidates == ["T1", "T3"]
