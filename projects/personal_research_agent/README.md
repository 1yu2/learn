# Personal Research Agent

This is the final project for the learning plan.

## Goal

Build an AgentScope agent that accepts a research topic, creates a plan, reads
local materials, retrieves relevant context with RAG, and writes a concise
research summary.

## Initial Milestones

1. Define the agent prompt and safety constraints; exit when paths and writes are explicitly bounded.
2. Add a read-only local file tool and offline rejection tests.
3. Add a small local knowledge base and a reproducible retrieval query.
4. Add planning support and show task state transitions.
5. Stream events for observability and record one recoverable failure.
6. Add a structured summary schema with source references.
7. Complete the scoring checklist in `docs/capstone-evaluation.md`.

## Acceptance reference

Use [`docs/capstone-evaluation.md`](../../docs/capstone-evaluation.md) for the MVP boundary, required risk tests, demo checklist, and 100-point completion rubric.
