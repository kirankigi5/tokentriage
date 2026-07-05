"""Run ADK LlmAgents synchronously on local models.

Bridges the deterministic orchestrator to genuine Google ADK agent execution:
given an ADK LlmAgent and a prompt, it spins up an in-memory Runner, sends the
prompt, and returns the agent's final text response.

Used when TOKENTRIAGE_USE_ADK=1 (and by `tokentriage adk-demo`), so the Triage
and Verifier agents really execute through ADK + LiteLLM on the local Ollama
models — not just the fast direct HTTP path. This is what makes the
"multi-agent system on ADK" claim real and demonstrable.
"""
from __future__ import annotations

import asyncio

from google.genai import types

_APP = "tokentriage"


async def _run_async(agent, prompt: str) -> str:
    from google.adk.runners import InMemoryRunner

    runner = InMemoryRunner(agent=agent, app_name=_APP)
    session = await runner.session_service.create_session(app_name=_APP, user_id="tt")
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])
    final = ""
    async for ev in runner.run_async(user_id="tt", session_id=session.id, new_message=msg):
        if ev.is_final_response() and ev.content and ev.content.parts:
            final = ev.content.parts[0].text or ""
    return final


def run_llm_agent(agent, prompt: str) -> str:
    """Execute an ADK LlmAgent and return its final text.

    Safe to call from both sync code (CLI, benchmark) and inside a running event
    loop (the FastAPI proxy) — in the latter case it runs on a worker thread so
    it never collides with the active loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run_async(agent, prompt))
    # Already inside an event loop: offload to a thread with its own loop.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(_run_async(agent, prompt))).result()
