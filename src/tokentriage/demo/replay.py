"""Loader for seeding the database with a precomputed judge trace."""

import json
import time
import uuid
from pathlib import Path

from tokentriage import db

def seed_judge_data(trace_file: str = "benchmarks/judge_trace.jsonl") -> None:
    """Read a JSONL trace and inject it directly into the database."""
    print("Initializing database...")
    db.init_db()

    print("Clearing existing traces for a clean demo state...")
    with db.conn() as c:
        c.execute("DELETE FROM decisions")
        c.execute("DELETE FROM conversations")
        c.execute("DELETE FROM conv_messages")
        c.execute("DELETE FROM budget")

    trace_path = Path(trace_file)
    if not trace_path.exists():
        print(f"Warning: {trace_file} not found. Cannot seed judge data.")
        return

    print(f"Loading traces from {trace_file}...")
    lines = trace_path.read_text().splitlines()
    
    # We want to space the timestamps out over the last few hours
    now = time.time()
    time_step = 3600 / max(1, len(lines)) # 1 hour spread

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        
        t = json.loads(line)
        simulated_ts = now - 3600 + (i * time_step)
        conv_id = t["conv_id"]
        task = t["task"]
        answer = t["answer"]

        # Insert into conversations
        title = task[:60]
        with db.conn() as c:
            c.execute(
                """INSERT INTO conversations (id, created_at, updated_at, title)
                   VALUES (?,?,?,?)""",
                (conv_id, simulated_ts, simulated_ts, title)
            )
            
            # User message
            c.execute(
                "INSERT INTO conv_messages (conversation_id, ts, role, content, extra) VALUES (?,?,?,?,?)",
                (conv_id, simulated_ts, "user", task, None)
            )
            
            # Assistant message + routing extra
            extra = {
                "tokentriage": {
                    "task_type": t["task_type"],
                    "complexity": t["complexity"],
                    "tier": t["chosen_tier"],
                    "cost": t["cost_usd"],
                    "rationale": t["rationale"],
                    "cache_hit": bool(t["cache_hit"]),
                    "verdict": t["verdict"] if t["verified"] else None,
                    "escalated_to": t["escalated_to"],
                    "latency": t["dispatch_latency_ms"]
                }
            }
            c.execute(
                "INSERT INTO conv_messages (conversation_id, ts, role, content, extra) VALUES (?,?,?,?,?)",
                (conv_id, simulated_ts + (t["dispatch_latency_ms"]/1000.0), "assistant", answer, json.dumps(extra))
            )

            # Insert into decisions
            c.execute(
                """INSERT INTO decisions (ts, task_id, task_preview, task_type, complexity,
                   chosen_tier, rationale, cost_usd, baseline_cost_usd, cache_hit,
                   verified, verdict, escalated_to, dispatch_latency_ms)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (simulated_ts, str(uuid.uuid4()), task[:120], t["task_type"], t["complexity"],
                 t["chosen_tier"], t["rationale"], t["cost_usd"], t["baseline_cost_usd"],
                 t["cache_hit"], t["verified"], t["verdict"] if t["verified"] else None, 
                 t["escalated_to"], t["dispatch_latency_ms"])
            )

            # Budget spend
            if t["cost_usd"] > 0:
                c.execute(
                    "INSERT INTO budget (ts, tier, cost_usd, latency_ms) VALUES (?,?,?,?)",
                    (simulated_ts, t["chosen_tier"], t["cost_usd"], t["dispatch_latency_ms"])
                )

    print(f"Successfully seeded {len(lines)} trace items.")
