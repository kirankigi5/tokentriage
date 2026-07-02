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

import asyncio
import json
import queue
import threading

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

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


def _tt_meta(result) -> dict:
    """The TokenTriage routing receipt returned to clients (both endpoints)."""
    return {
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
    }


@app.post("/v1/route/stream")
async def route_stream(request: Request):
    """Server-Sent Events: stream each routing-pipeline stage live as it runs,
    so the UI can show what's happening under the hood in real time."""
    body = await request.json()
    messages = body.get("messages", [])
    user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
    task = user_turns[-1] if user_turns else ""
    client_key = request.headers.get("authorization", "anonymous")

    q: queue.Queue = queue.Queue()

    def worker():
        try:
            clean = gateway_check(task, client_key, _policy, _limiter)
            sent = [dict(m) for m in messages]
            for m in reversed(sent):
                if m.get("role") == "user":
                    m["content"] = clean
                    break
            result = route(clean, _policy, _cache, messages=sent or None,
                           on_event=lambda stage, detail: q.put({"stage": stage, **detail}))
            q.put({"stage": "RESULT", "answer": result.answer,
                   "tokentriage": _tt_meta(result)})
        except SecurityError as e:
            q.put({"stage": "QUARANTINE", "detail": e.reason})
        except Exception as e:  # surface failures to the client instead of hanging
            q.put({"stage": "ERROR", "detail": str(e)})
        q.put(None)  # sentinel: stream complete

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


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
        "tokentriage": _tt_meta(result),
    }


@app.get("/conversations")
def conversations_list():
    """List saved conversations (most recent first)."""
    return db.list_conversations()


@app.post("/conversations")
async def conversations_save(request: Request):
    """Persist a conversation server-side so it survives across devices/reloads."""
    body = await request.json()
    conv_id = body.get("id")
    if not conv_id:
        return JSONResponse(status_code=400, content={"error": "id required"})
    db.save_conversation(conv_id, body.get("messages", []), body.get("title"))
    return {"ok": True, "id": conv_id}


@app.get("/conversations/{conv_id}")
def conversations_get(conv_id: str):
    return {"id": conv_id, "messages": db.get_conversation(conv_id)}


@app.delete("/conversations/{conv_id}")
def conversations_delete(conv_id: str):
    db.delete_conversation(conv_id)
    return {"ok": True}


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
