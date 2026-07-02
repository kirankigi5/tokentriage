"""Context privacy policy: what conversation history reaches a CLOUD tier.

Phase 1 runs entirely local, so these tests exercise the cloud path directly
with a synthetic cloud tier — the guarantee must hold the moment a cloud tier
is enabled.
"""
from tokentriage.agents.orchestrator import _privacy_context
from tokentriage.models.registry import ModelTier

LOCAL = ModelTier("T1", "qwen2.5:3b", 0.02, 0.05, "", "ollama")
CLOUD = ModelTier("T3", "gemini-2.5-pro", 1.25, 10.0, "", "gemini")

CONVO = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "What are the tax implications of that?"},  # sensitive
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "And its population?"},  # current turn
]
TASK = "And its population?"


def test_local_tier_gets_full_context_unchanged():
    msgs, note = _privacy_context(LOCAL, TASK, CONVO, {})
    assert msgs == CONVO and note is None  # on-device: nothing withheld


def test_cloud_none_sends_only_current_turn():
    pol = {"privacy": {"cloud_context": "none", "sensitive_firewall": False}}
    msgs, note = _privacy_context(CLOUD, TASK, CONVO, pol)
    assert msgs == [{"role": "user", "content": TASK}]
    assert "none" in note


def test_cloud_firewall_strips_sensitive_prior_turns():
    pol = {"privacy": {"cloud_context": "full", "sensitive_firewall": True}}
    msgs, note = _privacy_context(CLOUD, TASK, CONVO, pol)
    joined = " ".join(m["content"] for m in msgs)
    assert "tax" not in joined                 # sensitive prior turn removed
    assert "Paris" in joined                    # non-sensitive prior turn kept
    assert msgs[-1]["content"] == TASK          # current turn preserved
    assert "firewall" in note


def test_cloud_last_n_limits_history():
    pol = {"privacy": {"cloud_context": "last_n", "context_last_n": 1,
                       "sensitive_firewall": False}}
    msgs, note = _privacy_context(CLOUD, TASK, CONVO, pol)
    assert msgs[-1]["content"] == TASK
    assert len(msgs) <= 3                        # ~1 turn of history + current
    assert "last_1" in note
