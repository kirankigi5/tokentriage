"""SQLite persistence layer.

One small DB, five tables:
  decisions   — every routing decision with full trace (feeds dashboard + report)
  feedback    — (task_type, tier, verdict) tuples; `tokentriage tune` learns from these
  benchmarks  — per (model, task_type) measured accuracy; consumed via MCP tool
  budget      — spend ledger; the circuit breaker reads daily totals here
  cache       — semantic cache entries (embedding stored as JSON list)

Deliberately SQLite: zero infra for judges to reproduce, and the whole state
of the system is one inspectable file.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import contextmanager

from tokentriage.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    task_id TEXT NOT NULL,
    task_preview TEXT,          -- first 120 chars, for the dashboard feed
    task_type TEXT,
    complexity REAL,
    chosen_tier TEXT,
    rationale TEXT,
    cost_usd REAL,
    baseline_cost_usd REAL,     -- what always-Pro would have cost
    cache_hit INTEGER DEFAULT 0,
    verified INTEGER DEFAULT 0, -- 0 = not sampled, 1 = sampled
    verdict TEXT,               -- pass / fail / NULL
    escalated_to TEXT,          -- tier it escalated to, if any
    dispatch_latency_ms REAL DEFAULT 0,
    error TEXT                  -- transient or permanent error message, if any
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    task_type TEXT NOT NULL,
    tier TEXT NOT NULL,
    verdict TEXT NOT NULL       -- pass / fail
);
CREATE TABLE IF NOT EXISTS benchmarks (
    model_tier TEXT NOT NULL,
    task_type TEXT NOT NULL,
    accuracy REAL NOT NULL,
    samples INTEGER DEFAULT 0,
    PRIMARY KEY (model_tier, task_type)
);
CREATE TABLE IF NOT EXISTS budget (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    tier TEXT NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    task TEXT NOT NULL,
    answer TEXT NOT NULL,
    embedding TEXT NOT NULL     -- JSON list[float]
);
CREATE TABLE IF NOT EXISTS quarantine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    task_preview TEXT,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    title TEXT,
    is_pinned INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS conv_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    extra TEXT               -- JSON of non-core fields (routing meta, latency)
);
"""

# Seed accuracies so routing works before any learning has happened.
# `tokentriage tune` overwrites these from real verifier feedback.
_SEED_BENCHMARKS = [
    # (tier, task_type, accuracy)
    ("T1", "factual_lookup", 0.96), ("T1", "classification", 0.93),
    ("T1", "creative_short", 0.90), ("T1", "summarization", 0.88),
    ("T1", "multi_step_reasoning", 0.61), ("T1", "code_generation", 0.70),
    ("T1", "legal_or_financial", 0.55),
    ("T2", "factual_lookup", 0.98), ("T2", "classification", 0.96),
    ("T2", "creative_short", 0.94), ("T2", "summarization", 0.95),
    ("T2", "multi_step_reasoning", 0.85), ("T2", "code_generation", 0.88),
    ("T2", "legal_or_financial", 0.80),
    ("T3", "factual_lookup", 0.99), ("T3", "classification", 0.98),
    ("T3", "creative_short", 0.97), ("T3", "summarization", 0.98),
    ("T3", "multi_step_reasoning", 0.96), ("T3", "code_generation", 0.95),
    ("T3", "legal_or_financial", 0.94),
]


@contextmanager
def conn():
    c = sqlite3.connect(settings.db_path)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    with conn() as c:
        c.executescript(_SCHEMA)
        try:
            c.execute("ALTER TABLE conversations ADD COLUMN is_pinned INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            c.execute("ALTER TABLE decisions ADD COLUMN dispatch_latency_ms REAL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            c.execute("ALTER TABLE decisions ADD COLUMN error TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            c.execute("ALTER TABLE budget ADD COLUMN latency_ms REAL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
        for tier, tt, acc in _SEED_BENCHMARKS:
            c.execute(
                "INSERT OR IGNORE INTO benchmarks (model_tier, task_type, accuracy, samples)"
                " VALUES (?, ?, ?, 0)",
                (tier, tt, acc),
            )


def log_decision(**kw) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO decisions (ts, task_id, task_preview, task_type, complexity,
               chosen_tier, rationale, cost_usd, baseline_cost_usd, cache_hit,
               verified, verdict, escalated_to, dispatch_latency_ms, error)
               VALUES (:ts,:task_id,:task_preview,:task_type,:complexity,:chosen_tier,
               :rationale,:cost_usd,:baseline_cost_usd,:cache_hit,:verified,:verdict,
               :escalated_to,:dispatch_latency_ms,:error)""",
            {"ts": time.time(), "verdict": None, "escalated_to": None,
             "verified": 0, "cache_hit": 0, "dispatch_latency_ms": 0.0, "error": None, **kw},
        )


def record_feedback(task_type: str, tier: str, verdict: str) -> None:
    with conn() as c:
        c.execute("INSERT INTO feedback (ts, task_type, tier, verdict) VALUES (?,?,?,?)",
                  (time.time(), task_type, tier, verdict))


def record_spend(tier: str, cost_usd: float, latency_ms: float = 0.0) -> None:
    with conn() as c:
        c.execute("INSERT INTO budget (ts, tier, cost_usd, latency_ms) VALUES (?,?,?,?)",
                  (time.time(), tier, cost_usd, latency_ms))


def spend_today_usd() -> float:
    day_start = time.time() - (time.time() % 86400)
    with conn() as c:
        row = c.execute("SELECT COALESCE(SUM(cost_usd),0) s FROM budget WHERE ts >= ?",
                        (day_start,)).fetchone()
        return float(row["s"])


def get_benchmark(tier: str, task_type: str) -> float:
    with conn() as c:
        row = c.execute(
            "SELECT accuracy FROM benchmarks WHERE model_tier=? AND task_type=?",
            (tier, task_type)).fetchone()
        # Unknown task types get a conservative low score -> routes higher. Safe default.
        return float(row["accuracy"]) if row else 0.50


def set_benchmark(tier: str, task_type: str, accuracy: float, samples: int) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO benchmarks (model_tier, task_type, accuracy, samples)
               VALUES (?,?,?,?)
               ON CONFLICT(model_tier, task_type)
               DO UPDATE SET accuracy=excluded.accuracy, samples=excluded.samples""",
            (tier, task_type, accuracy, samples))


def quarantine(task_preview: str, reason: str) -> None:
    with conn() as c:
        c.execute("INSERT INTO quarantine (ts, task_preview, reason) VALUES (?,?,?)",
                  (time.time(), task_preview[:120], reason))


def save_conversation(conv_id: str, messages: list[dict], title: str | None = None) -> str:
    """Persist a whole conversation (full replace of its messages). Title defaults
    to the first user turn. Non-core fields (routing meta, latency) go in `extra`."""
    now = time.time()
    if not title:
        first = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        title = (first[:60] or "New chat")
    with conn() as c:
        c.execute(
            """INSERT INTO conversations (id, created_at, updated_at, title)
               VALUES (?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at""",
            (conv_id, now, now, title))
        c.execute("DELETE FROM conv_messages WHERE conversation_id=?", (conv_id,))
        for m in messages:
            extra = {k: v for k, v in m.items() if k not in ("role", "content")}
            c.execute(
                "INSERT INTO conv_messages (conversation_id, ts, role, content, extra)"
                " VALUES (?,?,?,?,?)",
                (conv_id, now, m.get("role", "user"), m.get("content", ""),
                 json.dumps(extra) if extra else None))
    return conv_id


def rename_conversation(conv_id: str, title: str) -> None:
    with conn() as c:
        c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))


def pin_conversation(conv_id: str, is_pinned: int) -> None:
    with conn() as c:
        c.execute("UPDATE conversations SET is_pinned = ? WHERE id = ?", (is_pinned, conv_id))


def list_conversations(limit: int = 50) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            """SELECT c.id, c.title, c.updated_at, c.is_pinned,
                      (SELECT COUNT(*) FROM conv_messages m WHERE m.conversation_id=c.id) AS n
               FROM conversations c ORDER BY c.is_pinned DESC, c.updated_at DESC LIMIT ?""",
            (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT role, content, extra FROM conv_messages"
            " WHERE conversation_id=? ORDER BY id", (conv_id,)).fetchall()
    msgs = []
    for r in rows:
        m = {"role": r["role"], "content": r["content"]}
        if r["extra"]:
            m.update(json.loads(r["extra"]))
        msgs.append(m)
    return msgs


def delete_conversation(conv_id: str) -> None:
    with conn() as c:
        c.execute("DELETE FROM conv_messages WHERE conversation_id=?", (conv_id,))
        c.execute("DELETE FROM conversations WHERE id=?", (conv_id,))


def stats(window_hours: float = 24.0) -> dict:
    """Aggregates for the dashboard and `tokentriage report`."""
    since = time.time() - window_hours * 3600
    with conn() as c:
        row = c.execute(
            """SELECT COUNT(*) n,
                      COALESCE(SUM(cost_usd),0) cost,
                      COALESCE(SUM(baseline_cost_usd),0) baseline,
                      COALESCE(SUM(cache_hit),0) cache_hits,
                      SUM(CASE WHEN escalated_to IS NOT NULL THEN 1 ELSE 0 END) escalations
               FROM decisions WHERE ts >= ?""", (since,)).fetchone()
        tiers = c.execute(
            "SELECT chosen_tier, COUNT(*) n FROM decisions WHERE ts >= ? GROUP BY chosen_tier",
            (since,)).fetchall()
        recent = c.execute(
            """SELECT ts, task_preview, task_type, chosen_tier, cost_usd,
                      dispatch_latency_ms, verdict, escalated_to
               FROM decisions ORDER BY id DESC LIMIT 15""").fetchall()
        
        quarantined = c.execute("SELECT COUNT(*) n FROM quarantine WHERE ts >= ?", (since,)).fetchone()["n"]
        task_types = c.execute(
            "SELECT task_type, COUNT(*) n FROM decisions WHERE ts >= ? GROUP BY task_type ORDER BY n DESC",
            (since,)).fetchall()
        verdicts = c.execute(
            "SELECT verdict, COUNT(*) n FROM decisions WHERE ts >= ? AND verdict IS NOT NULL GROUP BY verdict",
            (since,)).fetchall()
        lat_rows = c.execute(
            """SELECT dispatch_latency_ms FROM decisions
               WHERE ts >= ? AND dispatch_latency_ms IS NOT NULL
                     AND dispatch_latency_ms > 0
               ORDER BY dispatch_latency_ms""", (since,)).fetchall()

    baseline = row["baseline"] or 0.0
    cost = row["cost"] or 0.0
    latencies = [float(r["dispatch_latency_ms"]) for r in lat_rows]
    p95_index = max(0, min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)) if latencies else 0
    return {
        "requests": row["n"],
        "cost_usd": round(cost, 6),
        "baseline_usd": round(baseline, 6),
        "savings_pct": round(100 * (1 - cost / baseline), 1) if baseline > 0 else 0.0,
        "cache_hits": row["cache_hits"],
        "escalations": row["escalations"] or 0,
        "tier_utilization": {r["chosen_tier"]: r["n"] for r in tiers},
        "budget_spent_today": round(spend_today_usd(), 6),
        "recent": [dict(r) for r in recent],
        "quarantined": quarantined,
        "task_types": {r["task_type"]: r["n"] for r in task_types},
        "verifier_stats": {r["verdict"]: r["n"] for r in verdicts},
        "avg_dispatch_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p95_dispatch_latency_ms": round(latencies[p95_index], 1) if latencies else 0.0,
    }
