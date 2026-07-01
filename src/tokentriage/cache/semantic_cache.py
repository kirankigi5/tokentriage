"""Semantic Cache — Tier 0, the $0 model.

Before any paid model is called, the incoming task is embedded and compared
(cosine similarity) against previously answered tasks. Above the policy
threshold (default 0.95) the cached answer is returned at zero cost.

Why it earns its place: in realistic workloads (support queries repeat
constantly) the hit rate becomes a headline number — "N% of queries answered
at $0" — and it reframes the pool as four tiers where the cheapest is free.

Implementation is deliberately simple: embeddings in SQLite as JSON, brute
force cosine in numpy. Fine for demo scale; the writeup notes sqlite-vec /
a vector DB as the production path.
"""
from __future__ import annotations

import json
import time

import numpy as np

from tokentriage import db, providers


class SemanticCache:
    def __init__(self, policy: dict):
        cache_pol = policy.get("cache", {})
        self.threshold = float(cache_pol.get("similarity_threshold", 0.95))
        self.ttl_s = float(cache_pol.get("ttl_hours", 168)) * 3600

    def _embed(self, text: str) -> np.ndarray | None:
        # Embeddings come from the provider layer (local Ollama in Phase 1).
        # Returns None if no embedder is available -> cache simply disabled.
        values = providers.embed(text)
        if not values:
            return None
        vec = np.array(values, dtype=np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    def lookup(self, task: str) -> str | None:
        """Return a cached answer if a stored task is similar enough."""
        q = self._embed(task)
        if q is None:
            return None
        cutoff = time.time() - self.ttl_s
        with db.conn() as c:
            rows = c.execute(
                "SELECT answer, embedding FROM cache WHERE ts >= ?", (cutoff,)
            ).fetchall()
        best, best_sim = None, 0.0
        for r in rows:  # brute force is fine at demo scale
            v = np.array(json.loads(r["embedding"]), dtype=np.float32)
            sim = float(np.dot(q, v))  # both normalized -> dot == cosine
            if sim > best_sim:
                best, best_sim = r["answer"], sim
        return best if best_sim >= self.threshold else None

    def store(self, task: str, answer: str) -> None:
        v = self._embed(task)
        if v is None:
            return
        with db.conn() as c:
            c.execute("INSERT INTO cache (ts, task, answer, embedding) VALUES (?,?,?,?)",
                      (time.time(), task, answer, json.dumps(v.tolist())))
