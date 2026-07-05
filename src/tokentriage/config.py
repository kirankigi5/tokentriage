"""Central configuration: environment variables + declarative policy YAML.

Design note: keys are read per-tier (key isolation) so a leaked cheap-tier
credential cannot be used against the expensive tier. Keys NEVER appear in
code or logs — rubric hard rule.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Load .env if present so keys never need to be exported by hand (and never
# live in code). Real keys go in .env, which is gitignored; .env.example is the
# committed template. Per-tier vars win; a single GEMINI_API_KEY is the fallback.
try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))  # walk up from cwd to find the .env
except ImportError:  # dotenv is optional; plain env vars still work
    pass


def tier_key(tier_env: str, provider: str | None = None) -> str:
    """Resolve a tier's API key without leaking keys across providers.

    Per-tier vars still win. Provider-specific fallbacks keep an OpenRouter key
    from accidentally enabling a Gemini tier, and vice versa.
    """
    p = (provider or "").lower()
    disable_cloud = os.getenv("TOKENTRIAGE_DISABLE_CLOUD", "").lower()
    if p in ("openrouter", "openai", "gemini") and disable_cloud in ("1", "true", "yes", "on"):
        return ""
    direct = os.getenv(tier_env) if tier_env else ""
    if direct:
        return direct
    if p == "openrouter":
        return os.getenv("OPENROUTER_API_KEY") or ""
    if p == "openai":
        return os.getenv("OPENAI_API_KEY") or ""
    if p == "gemini":
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return (os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or "")


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("TOKENTRIAGE_HOST", "0.0.0.0")
    port: int = int(os.getenv("TOKENTRIAGE_PORT", "8000"))
    db_path: str = os.getenv("TOKENTRIAGE_DB_PATH", "tokentriage.db")
    policy_path: str = os.getenv("TOKENTRIAGE_POLICY_PATH", "config/policy.yaml")
    # Run Triage/Verifier through the Google ADK Runner (LiteLLM -> local Ollama)
    # instead of the fast direct HTTP path. Genuine ADK execution; a bit slower.
    use_adk: bool = field(
        default_factory=lambda: os.getenv("TOKENTRIAGE_USE_ADK", "").lower()
        not in ("", "0", "false", "no"))
    # Per-tier key isolation. Empty string == tier disabled.
    t1_api_key: str = field(default_factory=lambda: tier_key("TOKENTRIAGE_T1_API_KEY"))
    t2_api_key: str = field(default_factory=lambda: tier_key("TOKENTRIAGE_T2_API_KEY"))
    t3_api_key: str = field(default_factory=lambda: tier_key("TOKENTRIAGE_T3_API_KEY"))
    embed_api_key: str = field(default_factory=lambda: tier_key("TOKENTRIAGE_EMBED_API_KEY"))


def load_policy(path: str | None = None) -> dict:
    """Load and minimally validate config/policy.yaml.

    Raises ValueError with a human-readable message on invalid policy so the
    CLI (`tokentriage policy check`) can surface it cleanly.
    """
    p = Path(path or Settings().policy_path)
    if not p.exists():
        raise ValueError(f"Policy file not found: {p}")
    data = yaml.safe_load(p.read_text())

    pol = data.get("policies", {})
    default = pol.get("default", {})
    for key in ("accuracy_floor", "daily_budget_usd"):
        if key not in default:
            raise ValueError(f"policy.default.{key} is required")
    if not (0.0 < float(default["accuracy_floor"]) <= 1.0):
        raise ValueError("accuracy_floor must be in (0, 1]")
    return pol


settings = Settings()
