# AgentScope Learning Operations Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe environment configuration, a measurable learning roadmap, an AgentScope interview question bank, and capstone completion criteria.

**Architecture:** Keep configuration in a small immutable `Settings` dataclass under `src/agentscope_learn`. It reads process environment first and an optional local `.env` second, validates values without making network calls, and exposes only redacted summaries. Keep learning and evaluation content as Markdown documents linked from the existing README.

**Tech Stack:** Python 3.11, standard library `dataclasses`/`pathlib`, pytest, Markdown, AgentScope 2.x.

---

### Task 1: Add the safe configuration module

**Files:**
- Create: `src/agentscope_learn/config.py`
- Modify: `src/agentscope_learn/__init__.py`

- [ ] **Step 1: Define configuration behavior and parsing helpers**

Implement `Settings` with fields `model_provider`, `api_key`, `model_name`, `model_base_url`, `log_level`, `workspace`, `rag_data_dir`, `rag_collection`, and `enable_tracing`. Add `Settings.from_env(env: Mapping[str, str] | None = None, dotenv_path: Path | None = None)`. When `env` is omitted, copy `os.environ`; otherwise use the supplied mapping. Load simple `KEY=VALUE` pairs from `dotenv_path` only for keys absent from the supplied environment. Strip matching single or double quotes, ignore blank lines and comments, and reject malformed non-comment lines with `ValueError`.

Use defaults `dashscope`, `qwen-plus`, `INFO`, `./.workspace`, `./data/knowledge`, `agentscope_learning`, and `false`. Resolve relative paths against `dotenv_path.parent` when a dotenv path is supplied, otherwise against the current working directory. Accept providers `dashscope`, `openai`, `ollama`; accept log levels `DEBUG`, `INFO`, `WARNING`, `ERROR`; parse booleans from `1`, `true`, `yes`, `on`, `0`, `false`, `no`, `off`.

- [ ] **Step 2: Add credential validation and redacted output**

Implement `require_model_credentials()` so `dashscope` requires `DASHSCOPE_API_KEY`, `openai` requires `OPENAI_API_KEY`, and `ollama` requires a non-empty `MODEL_BASE_URL` defaulting to `http://localhost:11434/v1`. Raise `ConfigurationError` with the variable name and remediation text. Implement `redacted_summary()` returning JSON-safe primitive values; replace any key with `"***configured***"` or `"<missing>"`, and return paths as strings.

- [ ] **Step 3: Export the public configuration API**

Export `ConfigurationError` and `Settings` from `src/agentscope_learn/__init__.py` while retaining the existing learning-plan exports.

### Task 2: Test configuration offline

**Files:**
- Create: `tests/test_config.py`

- [ ] **Step 1: Test defaults and environment precedence**

Cover `Settings.from_env({})` defaults, environment overrides, `.env` fallback values, and the rule that an explicit environment value wins over `.env`.

- [ ] **Step 2: Test validation and credential requirements**

Cover unknown provider, invalid log level, invalid boolean, malformed dotenv line, missing DashScope/OpenAI credentials, and local Ollama credential behavior. Use `tmp_path` for dotenv fixtures and `monkeypatch` only for isolated environment cases.

- [ ] **Step 3: Test paths and redaction**

Cover relative path resolution from a dotenv file, absolute path preservation, and the guarantee that a sentinel key value does not occur in `redacted_summary()`.

### Task 3: Add environment template and guide

**Files:**
- Create: `.env.example`
- Create: `docs/environment.md`

- [ ] **Step 1: Add the copyable template**

Include safe defaults for `MODEL_PROVIDER=dashscope`, `DASHSCOPE_API_KEY=`, `MODEL_NAME=qwen-plus`, optional `MODEL_BASE_URL=`, `AGENTSCOPE_LOG_LEVEL=INFO`, `AGENTSCOPE_WORKSPACE=./.workspace`, `RAG_DATA_DIR=./data/knowledge`, `RAG_COLLECTION=agentscope_learning`, and `ENABLE_TRACING=false`.

- [ ] **Step 2: Document setup and provider switching**

Document `uv venv`, `uv pip install -e ".[dev]"`, copying `.env.example` to `.env`, the DashScope key requirement, OpenAI-compatible variables, Ollama local URL, config inspection through `Settings.from_env().redacted_summary()`, and offline verification commands. State explicitly that `.env` and keys must never be committed.

### Task 4: Write the learning roadmap and interview bank

**Files:**
- Create: `docs/learning-roadmap.md`
- Create: `docs/interview-question-bank.md`

- [ ] **Step 1: Define the ten-stage execution schedule**

For stages 1-10, record objective, reading, coding exercise, artifact, and evidence of mastery. Group stages into four checkpoints: fundamentals (1-3), tool execution and control (4-6), context and retrieval (7-8), production direction and capstone (9-10). Include a weekly cadence of read, implement, explain, test, and review.

- [ ] **Step 2: Define objective mastery gates**

Require a runnable artifact, a concept note, one deliberate failure and diagnosis, and a short oral/written explanation for each stage. Define stage pass as all four artifacts present and a score of at least 7/10 on that stage's questions; define overall readiness as all stages passed and at least 80% overall question score.

- [ ] **Step 3: Add categorized interview questions**

Include at least 24 questions across AgentScope architecture, Message/Event, Model/structured output, Agent/Tool/Toolkit, permission/HITL, Plan, Context/Middleware, RAG/Memory, service/team, and engineering safety. For each question include key points, a follow-up, and a 0/1/2 scoring rubric. Add a self-test sheet and remediation rule for missed questions.

### Task 5: Define capstone directions and completion scoring

**Files:**
- Create: `docs/capstone-evaluation.md`
- Modify: `projects/personal_research_agent/README.md`

- [ ] **Step 1: Compare candidate project directions**

Compare local research assistant, knowledge-base Q&A assistant, and code-repository explainer by learning coverage, data complexity, safety risk, and demo value. Select the local research assistant as the recommended first capstone because it exercises Plan, read-only tools, RAG, events, permissions, and structured output with local data.

- [ ] **Step 2: Define MVP, milestones, and evidence**

Specify topic input, plan creation, read-only file access, RAG retrieval, structured research summary, streamed event log, permission boundary, offline tests, and README demo commands. Define milestone exits and a failure checklist for hallucinated citations, unauthorized writes, missing session isolation, and opaque errors.

- [ ] **Step 3: Add a 100-point completion rubric**

Score functionality 35, AgentScope concept coverage 20, testing/reliability 20, security/permissions 15, observability/documentation 10. Define 90+ complete, 75-89 usable beta, 60-74 learning prototype, and below 60 incomplete; require all critical safety checks regardless of total score.

### Task 6: Link the new baseline and verify the repository

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add navigation links and configuration usage**

Add a concise “环境与验收” section near the existing environment section linking `docs/environment.md`, `docs/learning-roadmap.md`, `docs/interview-question-bank.md`, and `docs/capstone-evaluation.md`. Mention `Settings` as the reusable configuration entry point without duplicating the full documents.

- [ ] **Step 2: Run the focused tests**

Run `python3 -m pytest tests/test_config.py tests/test_learning_plan.py tests/test_custom_tool.py -q`. Expected: all tests pass.

- [ ] **Step 3: Run repository verification**

Run `python3 -m pytest -q`, `python3 -m compileall src examples tests`, and `git diff --check`. Expected: no test failures, compile errors, or whitespace errors. Confirm `git status --short` does not include `.env` or generated workspace data.
