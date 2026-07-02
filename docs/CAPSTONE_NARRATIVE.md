# TokenTriage Capstone Narrative

## Problem

Production AI agents waste money by sending every task to a frontier model,
even when a local or cheaper tier is sufficient. The result is invisible
overspend, weak governance, and no audit trail for why a model was selected.

## Solution

TokenTriage is an Agent FinOps control plane. It sits in front of agent apps as
an OpenAI-compatible and Gemini-compatible gateway, classifies each task, checks
semantic cache, asks MCP tools for benchmark/cost/budget data, applies policy,
dispatches to the cheapest sufficient model tier, and verifies sampled answers.

## Why Agents

The system uses a multi-agent pattern because routing quality is not just a
static if/else decision. The Triage Agent classifies work, the Verifier Agent
audits cheap-tier answers, and the deterministic orchestrator coordinates
policy, budget, cache, and escalation so cost control remains predictable.

## Why It Wins

- Enterprise pain: inference cost is now an operating expense.
- Drop-in adoption: existing apps can switch base URLs.
- Google fit: Gemini-compatible ingress and optional Gemini backend tiers.
- Evidence: `tokentriage evidence` creates reproducible reports.
- Security: injection quarantine, rate limits, budget breaker, and cloud-context
  privacy controls are built in.

## Demo Story

The Chainlit demo shows the full routing pipeline as visible steps. Judges see
security, cache, triage, MCP pricing, policy routing, dispatch, verification,
and savings receipt for every request.
