"""Provider dispatch layer — one generate() for every backend.

Each model tier (see models/registry.py) names a `provider`:
  "ollama"     -> local open-source model via the Ollama HTTP API ($0, Phase 1)
  "gemini"     -> Google Gemini API
  "openai"     -> OpenAI API
  "openrouter" -> OpenRouter's OpenAI-compatible API for cloud rescue tiers

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
import time
from functools import wraps

import httpx

# Ollama's local server. Override with OLLAMA_HOST if you run it elsewhere.
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# Local models (esp. 14B) can take a while on first load; be patient.
_OLLAMA_TIMEOUT = float(os.getenv("TOKENTRIAGE_OLLAMA_TIMEOUT", "180"))
_OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")

# Some open-source models (e.g. deepseek-r1) emit <think>...</think> traces.
# Strip them so answers and token accounting reflect the actual response.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _approx_tokens(text: str) -> int:
    """Fallback token estimate (~4 chars/token) when a backend omits counts."""
    return max(1, len(text) // 4)


def _approx_msg_tokens(messages: list[dict]) -> int:
    return _approx_tokens(" ".join(m.get("content", "") for m in messages))


def _reported_or_approx(value, fallback: int) -> int:
    """Use backend token counts only when they are positive.

    Some free/provider-routed endpoints report 0 usage. That is useful for
    billing, but not for our baseline savings receipt, so fall back to an
    estimate when the reported count is missing or zero.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return fallback
    return n if n > 0 else fallback


def _with_retries(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        backoff = 1.0
        last_err = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_err = e
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504):
                    last_err = e
                else:
                    raise
            except Exception as e:
                name = type(e).__name__
                if any(k in name for k in ("RateLimit", "Timeout", "InternalServer", "APIConnection")):
                    last_err = e
                else:
                    raise
            
            print(f"Retrying after error: {last_err}")
            time.sleep(backoff)
            backoff *= 2
        raise last_err
    return wrapper


@_with_retries
def _ollama_generate(model_id: str, messages: list[dict]) -> tuple[str, int, int]:
    """Call a local Ollama model with a full message list (multi-turn ready)."""
    r = httpx.post(
        f"{_OLLAMA_HOST}/api/chat",
        # temperature 0 -> greedy decoding, so benchmark runs are reproducible.
        json={"model": model_id, "messages": messages, "stream": False,
              "options": {"temperature": 0}},
        timeout=_OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    text = _THINK_RE.sub("", data.get("message", {}).get("content", "")).strip()
    itok = data.get("prompt_eval_count") or _approx_msg_tokens(messages)
    otok = data.get("eval_count") or _approx_tokens(text)
    return text, int(itok), int(otok)


def _to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Map OpenAI-style roles to Gemini's ('assistant' -> 'model')."""
    return [{"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m.get("content", "")}]} for m in messages]


@_with_retries
def _gemini_generate(model_id: str, api_key: str, messages: list[dict]) -> tuple[str, int, int]:
    """Call the Gemini API (Phase 2). Imported lazily so Phase 1 needs no key."""
    from google import genai
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model_id, contents=_to_gemini_contents(messages))
    text = resp.text or ""
    um = getattr(resp, "usage_metadata", None)
    itok = getattr(um, "prompt_token_count", None) or _approx_msg_tokens(messages)
    otok = getattr(um, "candidates_token_count", None) or _approx_tokens(text)
    return text, int(itok), int(otok)


@_with_retries
def _openai_generate(model_id: str, api_key: str, messages: list[dict]) -> tuple[str, int, int]:
    """Call the OpenAI API (Phase 3). Imported lazily; needs the openai package."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model_id, messages=messages)
    text = resp.choices[0].message.content or ""
    u = resp.usage
    itok = _reported_or_approx(getattr(u, "prompt_tokens", None), _approx_msg_tokens(messages))
    otok = _reported_or_approx(getattr(u, "completion_tokens", None), _approx_tokens(text))
    return text, int(itok), int(otok)


@_with_retries
def _openrouter_generate(model_id: str, api_key: str, messages: list[dict]) -> tuple[str, int, int]:
    """Call OpenRouter through the OpenAI-compatible chat completions API."""
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=_OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "TokenTriage"),
        },
    )
    resp = client.chat.completions.create(model=model_id, messages=messages)
    text = resp.choices[0].message.content or ""
    u = resp.usage
    itok = _reported_or_approx(getattr(u, "prompt_tokens", None), _approx_msg_tokens(messages))
    otok = _reported_or_approx(getattr(u, "completion_tokens", None), _approx_tokens(text))
    return text, int(itok), int(otok)


def generate(tier, prompt: str, messages: list[dict] | None = None) -> tuple[str, int, int]:
    """Dispatch a generation to whatever backend the tier is configured for.

    `tier` is a ModelTier (has .provider, .model_id, .api_key). Pass `messages`
    (OpenAI-style role/content list) for multi-turn context; otherwise `prompt`
    is sent as a single user turn.
    """
    msgs = messages or [{"role": "user", "content": prompt}]
    if tier.provider == "ollama":
        return _ollama_generate(tier.model_id, msgs)
    if tier.provider == "gemini":
        return _gemini_generate(tier.model_id, tier.api_key, msgs)
    if tier.provider == "openai":
        return _openai_generate(tier.model_id, tier.api_key, msgs)
    if tier.provider == "openrouter":
        return _openrouter_generate(tier.model_id, tier.api_key, msgs)
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
