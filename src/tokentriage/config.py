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


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("TOKENTRIAGE_HOST", "0.0.0.0")
    port: int = int(os.getenv("TOKENTRIAGE_PORT", "8000"))
    db_path: str = os.getenv("TOKENTRIAGE_DB_PATH", "tokentriage.db")
    policy_path: str = os.getenv("TOKENTRIAGE_POLICY_PATH", "config/policy.yaml")
    # Per-tier key isolation. Empty string == tier disabled.
    t1_api_key: str = field(default_factory=lambda: os.getenv("TOKENTRIAGE_T1_API_KEY", ""))
    t2_api_key: str = field(default_factory=lambda: os.getenv("TOKENTRIAGE_T2_API_KEY", ""))
    t3_api_key: str = field(default_factory=lambda: os.getenv("TOKENTRIAGE_T3_API_KEY", ""))
    embed_api_key: str = field(default_factory=lambda: os.getenv("TOKENTRIAGE_EMBED_API_KEY", ""))


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
