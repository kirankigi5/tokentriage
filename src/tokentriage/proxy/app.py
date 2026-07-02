"""FastAPI Gateway — the deployability story.

Exposes an OpenAI-compatible POST /v1/chat/completions so any existing app
becomes a TokenTriage client by changing only its base_url. Also serves the
live dashboard (/dashboard) and a stats API (/api/stats) that feeds it.

Run: `tokentriage serve`  (uvicorn under the hood)
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from tokentriage import db
from tokentriage.agents.orchestrator import route
from tokentriage.cache.semantic_cache import SemanticCache
from tokentriage.config import load_policy
from tokentriage.mcp_server import tools as mcp_tools
from tokentriage.models.registry import TIERS
from tokentriage.security.gateway import RateLimiter, SecurityError, gateway_check

app = FastAPI(title="TokenTriage — Inference Cost Engine")

_policy: dict = {}
_cache: SemanticCache | None = None
_limiter = RateLimiter()


@app.on_event("startup")
def _startup() -> None:
    global _policy, _cache
    db.init_db()
    _policy = load_policy()
    _cache = SemanticCache(_policy)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible entrypoint. The client's `model` field is ignored —
    TokenTriage decides the model; that's the whole point."""
    body = await request.json()
    messages = body.get("messages", [])
    user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
    task = user_turns[-1] if user_turns else ""   # route on the LATEST user turn
    client_key = request.headers.get("authorization", "anonymous")

    try:
        task = gateway_check(task, client_key, _policy, _limiter)   # security first
        # Send the full conversation to the model for context, but reflect the
        # sanitized latest turn back in so the model never sees the raw input.
        sent = [dict(m) for m in messages]
        for m in reversed(sent):
            if m.get("role") == "user":
                m["content"] = task
                break
        result = route(task, _policy, _cache, messages=sent or None)
    except SecurityError as e:
        return JSONResponse(status_code=e.status,
                            content={"error": {"type": "tokentriage_security",
                                               "message": e.reason}})

    # OpenAI-shaped response; TokenTriage metadata rides in an extension field.
    return {
        "id": f"tt-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"tokentriage:{result.chosen_tier}",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.answer},
            "finish_reason": "stop",
        }],
        "tokentriage": {
            "task_id": result.task_id,
            "chosen_tier": result.chosen_tier,
            "model_id": TIERS[result.chosen_tier].model_id if result.chosen_tier in TIERS else result.chosen_tier,
            "task_type": result.task_type,
            "complexity": result.complexity,
            "rationale": result.rationale,
            "cost_usd": round(result.cost_usd, 6),
            "baseline_cost_usd": round(result.baseline_cost_usd, 6),
            "cache_hit": result.cache_hit,
            "verified": result.verified,
            "verdict": result.verdict,
            "escalated_to": result.escalated_to,
            "context_note": result.context_note,
            "trace": [{"state": s, "ts": ts} for s, ts in result.trace],
        },
    }


@app.get("/api/stats")
def api_stats(window_hours: float = 24.0):
    """Dashboard data source (polled every 2s by the page)."""
    return mcp_tools.get_routing_stats(window_hours)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html = Path(__file__).parent / "dashboard.html"
    return html.read_text()


@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
def chat():
    """The routing playground: type a message, watch the routing decision."""
    html = Path(__file__).parent / "chat.html"
    return html.read_text()


@app.get("/healthz")
def healthz():
    return {"ok": True}
