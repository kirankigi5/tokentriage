"""Model pool registry.

Eight tiers, cheapest first. Tier 0 is the semantic cache ($0). Prices are
USD per 1M tokens and are served to agents via the MCP server — agents never
hardcode prices, they *ask* for them (interoperability by design).

PHASE 1 (current): all tiers are local open-source models via Ollama, so the
whole pipeline runs offline for $0 and produces REAL inference + REAL token
counts. Local prices below are an *estimated compute cost* that scales with
model size (bigger model = more energy/time), NOT a bill — they exist so the
router can quantify the savings from choosing a smaller model, and so real
cloud prices drop straight in for Phase 2/3.

PHASE 2/3: point a tier at "gemini", "openai", or "openrouter" (set provider
+ model_id + api_key_env) and its real published price. Nothing above
providers.py changes.
"""
from __future__ import annotations

from dataclasses import dataclass

from tokentriage.config import settings


@dataclass(frozen=True)
class ModelTier:
    tier: str                 # "T0".."T7"
    model_id: str             # backend model id ("" for cache)
    input_usd_per_m: float    # $ / 1M input tokens  (estimated for local)
    output_usd_per_m: float   # $ / 1M output tokens (estimated for local)
    api_key_env: str          # env var holding this tier's key (cloud only)
    provider: str = "ollama"  # "ollama" | "gemini" | "openai" | "openrouter"

    @property
    def api_key(self) -> str:
        # Tier's own var first (isolation), then the shared-key fallback.
        from tokentriage.config import tier_key
        return tier_key(self.api_key_env, self.provider) if self.api_key_env else tier_key("", self.provider)


# Phase 1 ladder: local qwen2.5 family, then OpenRouter as an optional rescue.
# Swap any entry for a cloud tier by changing provider/model_id/prices, e.g.
#   "T3": ModelTier("T3", "gemini-2.5-flash", 0.30, 2.50, "TOKENTRIAGE_T3_API_KEY", "gemini")
TIERS: dict[str, ModelTier] = {
    "T0": ModelTier("T0", "semantic-cache", 0.0, 0.0, "", "cache"),
    "T1": ModelTier("T1", "qwen2.5:3b",  0.02, 0.05, "", "ollama"),
    "T2": ModelTier("T2", "qwen2.5:7b",  0.05, 0.12, "", "ollama"),
    "T3": ModelTier("T3", "qwen2.5:14b", 0.10, 0.25, "", "ollama"),
    # OpenRouter free rescue catalog. Each model can use its own isolated
    # OpenRouter key, falling back to OPENROUTER_API_KEY when blank.
    "T4": ModelTier("T4", "google/gemma-4-31b-it:free", 0.0, 0.0, "TOKENTRIAGE_OR_GEMMA_31B_KEY", "openrouter"),
    "T5": ModelTier("T5", "openai/gpt-oss-20b:free", 0.0, 0.0, "TOKENTRIAGE_OR_GPT_OSS_20B_KEY", "openrouter"),
    "T6": ModelTier("T6", "qwen/qwen3-next-80b-a3b-instruct:free", 0.0, 0.0, "TOKENTRIAGE_OR_QWEN_NEXT_80B_KEY", "openrouter"),
    "T7": ModelTier("T7", "qwen/qwen3-coder-480b-a35b:free", 0.0, 0.0, "TOKENTRIAGE_OR_QWEN_CODER_480B_KEY", "openrouter"),
}

# Ordered cheapest -> most expensive; routing walks this list upward.
TIER_ORDER = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]

# Cost baseline = the real alternative a business faces WITHOUT TokenTriage:
# sending EVERY task to a frontier cloud model. Savings are measured against
# this. It is never actually called in Phase 1 — it's the yardstick only.
# (gemini-2.5-pro published price, USD per 1M tokens; edit to your reference.)
BASELINE_REF = ModelTier("REF", "gemini-2.5-pro", 1.25, 10.00, "", "gemini")


def estimate_baseline_usd(input_tokens: int, output_tokens: int) -> float:
    """What this task WOULD cost sent to the cloud-frontier baseline model."""
    return (input_tokens * BASELINE_REF.input_usd_per_m
            + output_tokens * BASELINE_REF.output_usd_per_m) / 1_000_000


def next_tier_up(tier: str) -> str | None:
    """Escalation helper: the next more-capable tier, or None at the top."""
    i = TIER_ORDER.index(tier)
    return TIER_ORDER[i + 1] if i + 1 < len(TIER_ORDER) else None


def estimate_cost_usd(tier: str, input_tokens: int, output_tokens: int) -> float:
    t = TIERS[tier]
    return (input_tokens * t.input_usd_per_m + output_tokens * t.output_usd_per_m) / 1_000_000
