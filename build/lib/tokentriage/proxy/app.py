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
import os
import queue
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tokentriage import db
from tokentriage.agents.orchestrator import route
from tokentriage.cache.semantic_cache import SemanticCache
from tokentriage.config import load_policy
from tokentriage.evidence import JUDGE_SCENARIOS, seed_judge_replay
from tokentriage.mcp_server import tools as mcp_tools
from tokentriage.models.registry import TIERS
from tokentriage.security.gateway import RateLimiter, SecurityError, gateway_check

app = FastAPI(title="TokenTriage — Inference Cost Engine")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_IMGS_DIR = _REPO_ROOT / "imgs"
if _IMGS_DIR.exists():
    app.mount("/imgs", StaticFiles(directory=_IMGS_DIR), name="imgs")

_policy: dict = {}
_cache: SemanticCache | None = None
_limiter = RateLimiter()
_demo_index = -1
_judge_mode = False


def _decision_rows() -> list[dict]:
    """Return decisions ordered by timestamp as a list of dicts."""
    with db.conn() as c:
        rows = c.execute(
            """SELECT id, ts, task_id, task_preview, task_type, complexity,
                      chosen_tier, rationale, cost_usd, baseline_cost_usd,
                      cache_hit, verified, verdict, escalated_to,
                      dispatch_latency_ms, error
               FROM decisions ORDER BY ts"""
        ).fetchall()
    return [dict(r) for r in rows]


def _conversation_for_task_preview(preview: str) -> list[dict] | None:
    """Try to find a conversation whose first message content matches the preview."""
    with db.conn() as c:
        row = c.execute(
            "SELECT conversation_id FROM conv_messages WHERE role='user' AND content LIKE ? ORDER BY id LIMIT 1",
            (preview[:120] + '%',),
        ).fetchone()
        if not row:
            return None
        conv_id = row["conversation_id"]
    return db.get_conversation(conv_id)


def _model_id_for_replay(tier: str) -> str:
    return {
        "T0": "semantic-cache",
        "T1": "qwen2.5:3b",
        "T2": "qwen2.5:7b",
        "T3": "qwen2.5:14b",
        "T4": "google/gemma-4-31b-it:free",
        "T5": "openai/gpt-oss-20b:free",
        "T6": "qwen/qwen3-next-80b-a3b-instruct:free",
        "T7": "qwen/qwen3-coder-480b-a35b:free",
    }.get(tier, "semantic-cache")


def _replay_meta(idx: int, row: dict) -> dict:
    return {
        "task_id": f"judge-{idx + 1:02d}",
        "chosen_tier": row.get("chosen_tier", ""),
        "model_id": _model_id_for_replay(row.get("chosen_tier", "")),
        "task_type": row.get("task_type", "replay"),
        "complexity": 0.1 if row.get("task_type") in ("factual_lookup", "cached") else 0.7,
        "rationale": row.get("rationale", "judge replay route"),
        "cost_usd": row.get("cost_usd", 0.0),
        "baseline_cost_usd": row.get("baseline_cost_usd", 0.0),
        "cache_hit": row.get("cache_hit", False),
        "verified": row.get("verified", False),
        "verdict": row.get("verdict"),
        "escalated_to": row.get("escalated_to"),
        "context_note": row.get("context_note"),
        "dispatch_latency_ms": row.get("dispatch_latency_ms", 0.0),
    }


def _replay_item(idx: int) -> dict:
    row = JUDGE_SCENARIOS[idx]
    meta = _replay_meta(idx, row)
    decision = {
        "id": idx + 1,
        "task_id": meta["task_id"],
        "task_preview": row["task"][:120],
        "task_type": row.get("task_type", "replay"),
        "complexity": meta["complexity"],
        "chosen_tier": row.get("chosen_tier", ""),
        "rationale": row.get("rationale", "judge replay route"),
        "cost_usd": row.get("cost_usd", 0.0),
        "baseline_cost_usd": row.get("baseline_cost_usd", 0.0),
        "cache_hit": int(bool(row.get("cache_hit", False))),
        "verified": int(bool(row.get("verified", False))),
        "verdict": row.get("verdict"),
        "escalated_to": row.get("escalated_to"),
        "dispatch_latency_ms": row.get("dispatch_latency_ms", 0.0),
        "error": None,
    }
    assistant = {
        "role": "assistant",
        "content": row.get("answer", ""),
        "events": row.get("events", []),
        "meta": meta,
        "ms": row.get("dispatch_latency_ms", 0.0) + 45,
    }
    return {
        "index": idx,
        "count": len(JUDGE_SCENARIOS),
        "decision": decision,
        "conversation": [
            {"role": "user", "content": row["task"]},
            assistant,
        ],
    }


@app.on_event("startup")
def _startup() -> None:
    global _policy, _cache, _judge_mode
    db.init_db()
    _policy = load_policy()
    _cache = SemanticCache(_policy)
    _judge_mode = os.getenv("TOKENTRIAGE_JUDGE_MODE", "").lower() in ("1", "true", "yes")


def _tt_meta(result) -> dict:
    """The TokenTriage routing receipt returned to clients (both endpoints)."""
    tier = TIERS.get(result.chosen_tier)
    return {
        "task_id": result.task_id,
        "chosen_tier": result.chosen_tier,
        "model_id": tier.model_id if tier else result.chosen_tier,
        "provider": tier.provider if tier else "",
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
        "dispatch_latency_ms": round(getattr(result, "dispatch_latency_ms", 0.0), 1),
        "trace": [{"state": s, "ts": ts} for s, ts in result.trace],
    }


def _messages_from_gemini(body: dict) -> list[dict]:
    """Translate Gemini generateContent input into our internal chat shape."""
    messages: list[dict] = []
    system = body.get("systemInstruction")
    if system:
        text = _parts_text(system.get("parts", []))
        if text:
            messages.append({"role": "system", "content": text})
    for item in body.get("contents", []):
        role = "assistant" if item.get("role") == "model" else "user"
        text = _parts_text(item.get("parts", []))
        if text:
            messages.append({"role": role, "content": text})
    return messages


def _parts_text(parts: list[dict]) -> str:
    return "\n".join(str(p.get("text", "")) for p in parts if p.get("text"))


def _latest_user_turn(messages: list[dict]) -> str:
    user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
    return user_turns[-1] if user_turns else ""


def _sanitize_latest_user(messages: list[dict], task: str) -> list[dict]:
    sent = [dict(m) for m in messages]
    for m in reversed(sent):
        if m.get("role") == "user":
            m["content"] = task
            break
    return sent


@app.post("/v1/route/stream")
async def route_stream(request: Request):
    """Server-Sent Events: stream each routing-pipeline stage live as it runs,
    so the UI can show what's happening under the hood in real time."""
    body = await request.json()
    messages = body.get("messages", [])
    task = _latest_user_turn(messages)
    client_key = request.headers.get("authorization", "anonymous")

    q: queue.Queue = queue.Queue()

    def worker():
        try:
            clean = gateway_check(task, client_key, _policy, _limiter)
            sent = _sanitize_latest_user(messages, clean)
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
    task = _latest_user_turn(messages)   # route on the LATEST user turn
    client_key = request.headers.get("authorization", "anonymous")

    try:
        task = gateway_check(task, client_key, _policy, _limiter)   # security first
        # Send the full conversation to the model for context, but reflect the
        # sanitized latest turn back in so the model never sees the raw input.
        sent = _sanitize_latest_user(messages, task)
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


@app.post("/v1beta/models/{model}:generateContent")
async def gemini_generate_content(model: str, request: Request):
    """Gemini-compatible ingress. The requested model is advisory: TokenTriage
    still selects the cheapest sufficient tier and returns Gemini-shaped output."""
    body = await request.json()
    messages = _messages_from_gemini(body)
    task = _latest_user_turn(messages)
    client_key = request.headers.get("x-goog-api-key") or request.headers.get(
        "authorization", "anonymous")

    try:
        task = gateway_check(task, client_key, _policy, _limiter)
        sent = _sanitize_latest_user(messages, task)
        result = route(task, _policy, _cache, messages=sent or None)
    except SecurityError as e:
        return JSONResponse(
            status_code=e.status,
            content={"error": {"code": e.status, "message": e.reason,
                               "status": "INVALID_ARGUMENT"}},
        )

    meta = _tt_meta(result)
    return {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": result.answer}],
            },
            "finishReason": "STOP",
            "index": 0,
        }],
        "modelVersion": f"tokentriage:{result.chosen_tier}:{meta['model_id']}",
        "responseId": f"tt-{uuid.uuid4().hex[:12]}",
        "tokentriage": meta,
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


@app.patch("/conversations/{conv_id}")
async def conversations_rename(conv_id: str, request: Request):
    body = await request.json()
    if not body.get("title"):
        return JSONResponse(status_code=400, content={"error": "title required"})
    db.rename_conversation(conv_id, body["title"])
    return {"ok": True}


@app.patch("/conversations/{conv_id}/pin")
async def conversations_pin(conv_id: str, request: Request):
    body = await request.json()
    db.pin_conversation(conv_id, int(bool(body.get("is_pinned"))))
    return {"ok": True}


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


def _ensure_judge_mode():
    if not _judge_mode:
        raise HTTPException(status_code=404, detail="demo replay unavailable")


@app.post("/demo/replay/reset")
def demo_replay_reset():
    """Re-seed the polished deterministic judge replay and reset the in-memory pointer."""
    _ensure_judge_mode()
    seed_judge_replay()
    global _demo_index
    _demo_index = -1
    return {"ok": True, "message": "judge trace seeded", "items": len(JUDGE_SCENARIOS)}


@app.get("/demo/replay/list")
def demo_replay_list():
    """List deterministic replay scenarios in carousel order."""
    _ensure_judge_mode()
    return {
        "count": len(JUDGE_SCENARIOS),
        "index": _demo_index,
        "items": [
            {
                "idx": i,
                "id": i + 1,
                "task_preview": row["task"][:120],
                "task_type": row.get("task_type", "replay"),
                "chosen_tier": row.get("chosen_tier", ""),
                "blocked": bool(row.get("blocked", False)),
            }
            for i, row in enumerate(JUDGE_SCENARIOS)
        ],
    }


@app.get("/demo/replay/item/{idx}")
def demo_replay_item(idx: int):
    """Return the prerecorded replay item at index `idx`."""
    _ensure_judge_mode()
    if idx < 0 or idx >= len(JUDGE_SCENARIOS):
        return JSONResponse(status_code=404, content={"error": "index out of range"})
    return _replay_item(idx)


@app.get("/demo/replay/next")
def demo_replay_next():
    _ensure_judge_mode()
    global _demo_index
    if not JUDGE_SCENARIOS:
        return JSONResponse(status_code=404, content={"error": "no replay items seeded"})
    _demo_index = (_demo_index + 1) % len(JUDGE_SCENARIOS)
    return demo_replay_item(_demo_index)


@app.get("/demo/replay/prev")
def demo_replay_prev():
    _ensure_judge_mode()
    global _demo_index
    if not JUDGE_SCENARIOS:
        return JSONResponse(status_code=404, content={"error": "no replay items seeded"})
    _demo_index = (_demo_index - 1) % len(JUDGE_SCENARIOS)
    return demo_replay_item(_demo_index)


@app.get("/api/config")
def api_config():
    return {"judge_mode": _judge_mode}


@app.get("/architecture", response_class=HTMLResponse)
def architecture():
    html = Path(__file__).parent / "architecture.html"
    return html.read_text()


@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
def chat():
    """The routing playground: type a message, watch the routing decision."""
    html = Path(__file__).parent / "chat.html"
    return html.read_text()


@app.head("/")
def root_head():
    return {}


@app.get("/healthz")
def healthz():
    return {"ok": True}
