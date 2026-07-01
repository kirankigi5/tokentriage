"""TokenTriage MCP Server.

Exposes the six cost-intelligence tools over the Model Context Protocol so
any MCP-compatible agent can query pricing, benchmarks, budget, and stats.

Run standalone:
    python -m tokentriage.mcp_server.server          # stdio transport
or via the CLI:
    tokentriage serve --with-mcp

FastMCP import verified against mcp SDK >=1.0. For remote deployment,
switch to streamable-HTTP transport: mcp.run(transport="streamable-http").
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from tokentriage import db
from tokentriage.mcp_server import tools

mcp = FastMCP("tokentriage")


@mcp.tool()
def get_model_pricing(tier: str) -> dict:
    """Return USD-per-1M-token pricing for a model tier (T1/T2/T3)."""
    return tools.get_model_pricing(tier)


@mcp.tool()
def get_accuracy_benchmark(tier: str, task_type: str) -> float:
    """Return measured accuracy for a (tier, task_type) pair."""
    return tools.get_accuracy_benchmark(tier, task_type)


@mcp.tool()
def check_budget_remaining() -> dict:
    """Return today's spend cap, spend so far, and remaining budget in USD."""
    return tools.check_budget_remaining()


@mcp.tool()
def log_routing_decision(task_id: str, chosen_tier: str, reason: str, cost_usd: float) -> str:
    """Append a routing decision to the audit log."""
    return tools.log_routing_decision(task_id, chosen_tier, reason, cost_usd)


@mcp.tool()
def get_routing_stats(window_hours: float = 24.0) -> dict:
    """Return aggregate routing stats: cost, savings %, cache hits, tier usage."""
    return tools.get_routing_stats(window_hours)


@mcp.tool()
def quarantine_request(task_preview: str, reason: str) -> str:
    """Record a security-flagged request for audit."""
    return tools.quarantine_request(task_preview, reason)


if __name__ == "__main__":
    db.init_db()
    mcp.run()  # stdio transport by default
