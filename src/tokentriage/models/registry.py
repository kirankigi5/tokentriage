"""Model pool registry.

Four tiers, cheapest first. Tier 0 is the semantic cache ($0). Pricing is
USD per 1M tokens and is served to agents via the MCP server — agents never
hardcode prices, they *ask* for them (interoperability by design).

Model IDs and pricing verified against Gemini API docs (July 2026).
"""
from __future__ import annotations

from dataclasses import dataclass

from tokentriage.config import settings


@dataclass(frozen=True)
class ModelTier:
    tier: str                 # "T0".."T3"
    model_id: str             # Gemini model string ("" for cache)
    input_usd_per_m: float    # $ / 1M input tokens
    output_usd_per_m: float   # $ / 1M output tokens
    api_key_env: str          # which env var holds this tier's key (isolation)

    @property
    def api_key(self) -> str:
        import os
        return os.getenv(self.api_key_env, "")


TIERS: dict[str, ModelTier] = {
    "T0": ModelTier("T0", "semantic-cache", 0.0, 0.0, ""),
    "T1": ModelTier("T1", "gemini-3.1-flash-lite", 0.25, 1.50, "TOKENTRIAGE_T1_API_KEY"),
    "T2": ModelTier("T2", "gemini-3.5-flash", 1.50, 9.00, "TOKENTRIAGE_T2_API_KEY"),
    "T3": ModelTier("T3", "gemini-3.1-pro", 2.00, 12.00, "TOKENTRIAGE_T3_API_KEY"),
}

# Ordered cheapest -> most expensive; routing walks this list upward.
TIER_ORDER = ["T0", "T1", "T2", "T3"]


def next_tier_up(tier: str) -> str | None:
    """Escalation helper: the next more-capable tier, or None at the top."""
    i = TIER_ORDER.index(tier)
    return TIER_ORDER[i + 1] if i + 1 < len(TIER_ORDER) else None


def estimate_cost_usd(tier: str, input_tokens: int, output_tokens: int) -> float:
    t = TIERS[tier]
    return (input_tokens * t.input_usd_per_m + output_tokens * t.output_usd_per_m) / 1_000_000
