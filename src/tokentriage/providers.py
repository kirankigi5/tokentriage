"""Provider dispatch layer — one generate() for every backend.

Each model tier (see models/registry.py) names a `provider`:
  "ollama"  -> local open-source model via the Ollama HTTP API ($0, Phase 1)
  "gemini"  -> Google Gemini API              (Phase 2, free tier)
  "openai"  -> OpenAI API                     (Phase 3, paid)

This module turns (tier, prompt) into (text, input_tokens, output_tokens) so
the orchestrator, triage, and verifier stay provider-agnostic. Adding a cloud
provider later is a registry edit plus a branch here — nothing above this layer
changes. That is the whole point of routing behind an abstraction.

Token counts are the REAL counts each backend reports (Ollama:
prompt_eval_count / eval_count; Gemini: usage_metadata), so cost and savings
numbers are grounded in actual usage, not estimates.
"""
from __future__ import annotations

import os
import re

import httpx

# Ollama's local server. Override with OLLAMA_HOST if you run it elsewhere.
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# Local models (esp. 14B) can take a while on first load; be patient.
_OLLAMA_TIMEOUT = float(os.getenv("TOKENTRIAGE_OLLAMA_TIMEOUT", "180"))

# Some open-source models (e.g. deepseek-r1) emit <think>...</think> traces.
# Strip them so answers and token accounting reflect the actual response.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _approx_tokens(text: str) -> int:
    """Fallback token estimate (~4 chars/token) when a backend omits counts."""
    return max(1, len(text) // 4)


def _ollama_generate(model_id: str, prompt: str) -> tuple[str, int, int]:
    """Call a local Ollama model. Returns (text, input_tokens, output_tokens)."""
    r = httpx.post(
        f"{_OLLAMA_HOST}/api/chat",
        json={"model": model_id,
              "messages": [{"role": "user", "content": prompt}],
              "stream": False},
        timeout=_OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    text = _THINK_RE.sub("", data.get("message", {}).get("content", "")).strip()
    itok = data.get("prompt_eval_count") or _approx_tokens(prompt)
    otok = data.get("eval_count") or _approx_tokens(text)
    return text, int(itok), int(otok)


def _gemini_generate(model_id: str, api_key: str, prompt: str) -> tuple[str, int, int]:
    """Call the Gemini API (Phase 2). Imported lazily so Phase 1 needs no key."""
    from google import genai
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model_id, contents=prompt)
    text = resp.text or ""
    um = getattr(resp, "usage_metadata", None)
    itok = getattr(um, "prompt_token_count", None) or _approx_tokens(prompt)
    otok = getattr(um, "candidates_token_count", None) or _approx_tokens(text)
    return text, int(itok), int(otok)


def _openai_generate(model_id: str, api_key: str, prompt: str) -> tuple[str, int, int]:
    """Call the OpenAI API (Phase 3). Imported lazily; needs the openai package."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_id, messages=[{"role": "user", "content": prompt}])
    text = resp.choices[0].message.content or ""
    u = resp.usage
    itok = getattr(u, "prompt_tokens", None) or _approx_tokens(prompt)
    otok = getattr(u, "completion_tokens", None) or _approx_tokens(text)
    return text, int(itok), int(otok)


def generate(tier, prompt: str) -> tuple[str, int, int]:
    """Dispatch a generation to whatever backend the tier is configured for.

    `tier` is a ModelTier (has .provider, .model_id, .api_key).
    """
    if tier.provider == "ollama":
        return _ollama_generate(tier.model_id, prompt)
    if tier.provider == "gemini":
        return _gemini_generate(tier.model_id, tier.api_key, prompt)
    if tier.provider == "openai":
        return _openai_generate(tier.model_id, tier.api_key, prompt)
    raise ValueError(f"Unknown provider for tier {tier.tier!r}: {tier.provider!r}")


# --- Embeddings (for the T0 semantic cache) -------------------------------
_EMBED_PROVIDER = os.getenv("TOKENTRIAGE_EMBED_PROVIDER", "ollama")
_OLLAMA_EMBED_MODEL = os.getenv("TOKENTRIAGE_OLLAMA_EMBED_MODEL", "nomic-embed-text")


def embed(text: str) -> list[float] | None:
    """Return an embedding vector for the semantic cache, or None if unavailable.

    Phase 1 uses a local Ollama embedding model (no key). Set
    TOKENTRIAGE_EMBED_PROVIDER=gemini to use Gemini embeddings instead.
    """
    if _EMBED_PROVIDER == "ollama":
        try:
            r = httpx.post(f"{_OLLAMA_HOST}/api/embeddings",
                           json={"model": _OLLAMA_EMBED_MODEL, "prompt": text},
                           timeout=_OLLAMA_TIMEOUT)
            r.raise_for_status()
            return r.json().get("embedding")
        except Exception:
            return None  # cache simply disabled if the embedder isn't available
    if _EMBED_PROVIDER == "gemini":
        from google import genai
        from tokentriage.config import settings
        if not settings.embed_api_key:
            return None
        client = genai.Client(api_key=settings.embed_api_key)
        resp = client.models.embed_content(
            model="gemini-embedding-001", contents=text)
        return list(resp.embeddings[0].values)
    return None
