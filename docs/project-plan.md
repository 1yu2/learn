# AgentScope Learn Project Plan

This repository is a staged learning workspace for AgentScope 2.x. It should
stay small, runnable, and easy to extend stage by stage.

## Repository Layers

| Layer | Path | Responsibility |
| --- | --- | --- |
| Learning guide | `README.md` | Human-facing roadmap, official docs links, stage tasks |
| Project metadata | `pyproject.toml` | Python version, package metadata, optional dependencies, pytest config |
| Reusable code | `src/agentscope_learn/` | Structured learning metadata and future shared helpers |
| Stage examples | `examples/` | Small scripts aligned with README stages |
| Notes | `notes/` | Concept summaries, API differences, troubleshooting records |
| Final project | `projects/personal_research_agent/` | Capstone AgentScope project workspace |
| Tests | `tests/` | Offline checks for structure, metadata, and safe example helpers |

## Implementation Logic

1. Keep official AgentScope 2.x docs as the source of truth.
2. Add each learning stage as a small, independently runnable example.
3. Put reusable metadata and pure helper functions under `src/agentscope_learn/`.
4. Keep examples import-light so they can be compiled without API keys.
5. Use tests for offline guarantees: structure, metadata, and local helper behavior.
6. Avoid committing secrets, generated caches, vector-store data, or local `.env` files.

## Current Milestones

- Stage 1 starts with `examples/01_quickstart/hello_agent.py`.
- Stage 2 inspects message and event flow.
- Stage 3 introduces structured model output.
- Stage 4 introduces custom tools.
- Stages 5-9 provide scaffolds for permission, planning, context, RAG, service, and team concepts.
- Stage 10 grows into `projects/personal_research_agent/`.

## Verification

Run these commands before committing:

```bash
python3 -m pytest -q
python3 -m compileall src examples tests
git diff --check
```
