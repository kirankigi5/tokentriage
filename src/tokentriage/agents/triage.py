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
import re
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


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response, tolerating markdown
    fences or surrounding prose that small local models sometimes add."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


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
    from google.adk.models.lite_llm import LiteLlm

    triage_agent = LlmAgent(
        name="triage_agent",
        # LiteLLM runs this ADK agent on the local Ollama model — no cloud, no key.
        model=LiteLlm(model=f"ollama_chat/{TIERS['T2'].model_id}"),
        instruction=TRIAGE_INSTRUCTION,
        description="Scores task complexity and assigns a task taxonomy for routing.",
    )
except ImportError:  # keeps unit tests runnable without ADK/LiteLLM installed
    triage_agent = None


# Deterministic safety-net for sensitive domains. Defense-in-depth: even if the
# LLM classifier mislabels one of these, the term match forces the task to
# `legal_or_financial`, which policy pins to the top tier (min_tier: T3). This
# is what makes "sensitive tasks are never downgraded" a guarantee, not a hope.
# Kept intentionally specific so ordinary business tasks aren't over-escalated.
_SENSITIVE_TERMS = (
    "tax", "contract", "liability", "legal", "lawsuit", "attorney", "litigation",
    "compliance", "gdpr", "hipaa", "regulatory", "indemnif", "warranty",
    "recognizing revenue", "revenue recognition", "gaap", "sec filing",
    "securities", "audit", "medical", "diagnosis", "prescription", "patient",
)


def is_sensitive(text: str) -> bool:
    """True if the text contains a sensitive-domain term. Reused by the context
    privacy firewall so flagged content is never included in cloud requests."""
    t = (text or "").lower()
    return any(k in t for k in _SENSITIVE_TERMS)


def _sensitive_backstop(task: str, verdict: TriageVerdict) -> TriageVerdict:
    """Upgrade to legal_or_financial if a sensitive term appears — regardless of
    what the LLM classifier decided. No-op if already classified that way."""
    if verdict.task_type == "legal_or_financial":
        return verdict
    t = task.lower()
    hit = next((k for k in _SENSITIVE_TERMS if k in t), None)
    if hit is None:
        return verdict
    return TriageVerdict(
        complexity_score=max(verdict.complexity_score, 0.75),
        task_type="legal_or_financial",
        estimated_output_tokens=verdict.estimated_output_tokens,
        rationale=f"sensitive-term backstop matched {hit!r}; "
                  f"pinned to top tier (was {verdict.task_type})",
    )


# High-confidence keyword -> task_type rules. A deterministic guardrail for the
# obvious classes a small local classifier sometimes mislabels (e.g. tagging a
# "classify this ticket" prompt as factual_lookup). Checked most-specific first;
# only fires on unambiguous phrasing so it can't override correct LLM labels.
_TAXONOMY_RULES = (
    ("code_generation", ("write a python", "write python", "write a sql", "sql query",
                         "function that", "function to", "a decorator", "regular expression")),
    ("multi_step_reasoning", ("calculate", "compute", "step by step", "break-even",
                              "breakeven", "growth rate", "walk through",
                              "show your reasoning", "breakeven analysis")),
    ("classification", ("classify", "positive or negative", "urgent or not-urgent",
                        "urgent or not", "tag this", "categorize", "sentiment")),
    ("summarization", ("summarize", "summarise", "tl;dr", "key point", "in two sentences")),
)


def _taxonomy_hint(task: str) -> str | None:
    t = task.lower()
    for label, kws in _TAXONOMY_RULES:
        if any(k in t for k in kws):
            return label
    return None


def _backstop(task: str, verdict: TriageVerdict) -> TriageVerdict:
    """Deterministic guardrails over the LLM label: (1) sensitive domains pin to
    the top tier; (2) obvious task classes are corrected by keyword. Defense in
    depth — the small model classifies, deterministic rules catch its blind spots."""
    pinned = _sensitive_backstop(task, verdict)
    if pinned.task_type == "legal_or_financial":
        return pinned
    hint = _taxonomy_hint(task)
    if hint and hint != verdict.task_type:
        harder = hint in ("code_generation", "multi_step_reasoning")
        return TriageVerdict(
            complexity_score=max(verdict.complexity_score, 0.6 if harder else verdict.complexity_score),
            task_type=hint,
            estimated_output_tokens=verdict.estimated_output_tokens,
            rationale=f"taxonomy backstop -> {hint} (LLM said {verdict.task_type})",
        )
    return verdict


def triage(task: str) -> TriageVerdict:
    """Direct-call path used by the orchestrator's state machine.

    Uses the same model + instruction as the ADK agent; the ADK Runner path
    is wired in orchestrator.py. Falls back to a conservative default if the
    model returns malformed JSON (fail-safe: unknown => route higher). A
    deterministic sensitive-term backstop runs on every result.

    When TOKENTRIAGE_USE_ADK=1, the classification runs through the ADK
    triage_agent (Runner + LiteLLM + Ollama); otherwise the fast direct path.
    """
    from tokentriage.config import settings
    if settings.use_adk and triage_agent is not None:
        from tokentriage.agents.adk_runtime import run_llm_agent
        text = run_llm_agent(triage_agent, task)  # instruction is on the agent
    else:
        text, _, _ = providers.generate(
            TIERS["T2"], f"{TRIAGE_INSTRUCTION}\n\nTASK:\n{task}")
    try:
        d = extract_json(text)
        tt = d["task_type"] if d.get("task_type") in TASK_TYPES else "multi_step_reasoning"
        verdict = TriageVerdict(
            complexity_score=max(0.0, min(1.0, float(d["complexity_score"]))),
            task_type=tt,
            estimated_output_tokens=int(d.get("estimated_output_tokens", 300)),
            rationale=str(d.get("rationale", ""))[:200],
        )
    except Exception:
        # Fail-safe: if triage is unreadable, assume hard task -> higher tier.
        verdict = TriageVerdict(0.9, "multi_step_reasoning", 500,
                                "triage_parse_failure_conservative_default")
    return _backstop(task, verdict)
