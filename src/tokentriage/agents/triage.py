"""Triage Agent (ADK sub-agent).

Classifies each incoming task into a structured verdict:
    complexity_score (0-1), task_type, estimated_output_tokens, rationale.

Meta-point for the writeup: triage runs on a small, cheap LOCAL model (the mid
tier, qwen2.5:7b) — capable enough to classify reliably, cheap enough that the
overhead of deciding is a tiny fraction of what routing saves. (The 3B tier was
too weak at the taxonomy; 7B gives clean labels at negligible cost.)

ADK imports verified against google-adk >=1.0: LlmAgent and Agent are aliases;
constructor accepts name, model, instruction, description, sub_agents.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from tokentriage import providers
from tokentriage.models.registry import TIERS

# The closed set of task types. Must stay in sync with:
#   - db._SEED_BENCHMARKS (per-type accuracy rows)
#   - config/policy.yaml task_overrides
TASK_TYPES = [
    "factual_lookup", "classification", "summarization", "creative_short",
    "multi_step_reasoning", "code_generation", "legal_or_financial",
]

TRIAGE_INSTRUCTION = f"""You are a task triage classifier for an LLM routing system.
Given a user task, output ONLY a JSON object with exactly these keys:
  "complexity_score": float 0.0-1.0 (0 = trivial lookup, 1 = expert multi-step work)
  "task_type": one of {TASK_TYPES}
  "estimated_output_tokens": int (rough expected answer length)
  "rationale": one short sentence explaining the score
Classify anything involving law, contracts, tax, medical billing, or financial
advice as "legal_or_financial" regardless of apparent simplicity.
No prose. No markdown fences. JSON only."""


@dataclass(frozen=True)
class TriageVerdict:
    complexity_score: float
    task_type: str
    estimated_output_tokens: int
    rationale: str


# --- ADK agent definition -------------------------------------------------
# Registered as a sub-agent of the Orchestrator (see orchestrator.py).
try:
    from google.adk.agents import LlmAgent

    triage_agent = LlmAgent(
        name="triage_agent",
        model=TIERS["T2"].model_id,          # small local model does the triage
        instruction=TRIAGE_INSTRUCTION,
        description="Scores task complexity and assigns a task taxonomy for routing.",
    )
except ImportError:  # keeps unit tests runnable without ADK installed
    triage_agent = None


def triage(task: str) -> TriageVerdict:
    """Direct-call path used by the orchestrator's state machine.

    Uses the same model + instruction as the ADK agent; the ADK Runner path
    is wired in orchestrator.py. Falls back to a conservative default if the
    model returns malformed JSON (fail-safe: unknown => route higher).
    """
    text, _, _ = providers.generate(
        TIERS["T2"], f"{TRIAGE_INSTRUCTION}\n\nTASK:\n{task}")
    try:
        raw = text.strip().strip("`").removeprefix("json").strip()
        d = json.loads(raw)
        tt = d["task_type"] if d.get("task_type") in TASK_TYPES else "multi_step_reasoning"
        return TriageVerdict(
            complexity_score=max(0.0, min(1.0, float(d["complexity_score"]))),
            task_type=tt,
            estimated_output_tokens=int(d.get("estimated_output_tokens", 300)),
            rationale=str(d.get("rationale", ""))[:200],
        )
    except Exception:
        # Fail-safe: if triage is unreadable, assume hard task -> higher tier.
        return TriageVerdict(0.9, "multi_step_reasoning", 500,
                             "triage_parse_failure_conservative_default")
