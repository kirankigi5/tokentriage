"""Verifier Agent (ADK sub-agent) — the self-correction loop.

After a cheap-tier answer, a sampled fraction (policy: escalation.verify_sample_rate)
of (task, answer) pairs is judged by this agent running on the MID tier (T2):
strong enough to judge, cheap enough that verification doesn't eat the savings.

Fail -> the orchestrator escalates one tier up and re-verifies (bounded by
escalation.max_hops). Every verdict is recorded to the feedback store, which
`tokentriage tune` uses to adapt per-(tier, task_type) accuracy — so repeated
failures teach the router to stop under-routing that task type.

Tradeoff documented for the writeup: verification is SAMPLED, not universal.
At sample rate s and verifier cost v, overhead is s*v per request; with
s=0.25 and T2 pricing this stays well under the T3-vs-T1 price gap.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from tokentriage import providers
from tokentriage.models.registry import TIERS

VERIFIER_INSTRUCTION = """You are a strict answer-quality judge.
Given a TASK and an ANSWER, decide if the answer is complete, correct, and
responsive to the task. Output ONLY JSON:
  {"verdict": "pass" | "fail", "reason": "<one short sentence>"}
Judge correctness and completeness, not style. No markdown fences."""


@dataclass(frozen=True)
class VerifyResult:
    verdict: str   # "pass" | "fail"
    reason: str


# --- ADK agent definition -------------------------------------------------
try:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    verifier_agent = LlmAgent(
        name="verifier_agent",
        # Runs on the local mid-tier model via LiteLLM — genuine ADK, no cloud.
        model=LiteLlm(model=f"ollama_chat/{TIERS['T2'].model_id}"),
        instruction=VERIFIER_INSTRUCTION,
        description="Samples cheap-tier answers and fails ones that need escalation.",
    )
except ImportError:
    verifier_agent = None


def should_sample(policy: dict, chosen_tier: str, task: str | None = None) -> bool:
    """Only cheap-tier answers are sampled; T3 answers are trusted by design.

    Sampling is keyed off the task hash (not RNG) so which answers get verified
    — and therefore which escalate — is reproducible across benchmark runs."""
    if chosen_tier in ("T0", "T3"):
        return False
    rate = float(policy.get("escalation", {}).get("verify_sample_rate", 0.25))
    if task is not None:
        h = int(hashlib.md5(task.encode("utf-8")).hexdigest(), 16) % 1000 / 1000.0
        return h < rate
    return random.random() < rate


def verify(task: str, answer: str) -> VerifyResult:
    from tokentriage.config import settings
    prompt = f"TASK:\n{task}\n\nANSWER:\n{answer}"
    if settings.use_adk and verifier_agent is not None:
        from tokentriage.agents.adk_runtime import run_llm_agent
        text = run_llm_agent(verifier_agent, prompt)  # instruction is on the agent
    else:
        text, _, _ = providers.generate(
            TIERS["T2"], f"{VERIFIER_INSTRUCTION}\n\n{prompt}")
    try:
        from tokentriage.agents.triage import extract_json
        d = extract_json(text)
        v = d.get("verdict", "pass")
        return VerifyResult("fail" if v == "fail" else "pass",
                            str(d.get("reason", ""))[:200])
    except Exception:
        # Fail-safe: an unreadable verdict must never trigger a paid escalation.
        return VerifyResult("pass", "verifier_parse_failure_defaulted_pass")
